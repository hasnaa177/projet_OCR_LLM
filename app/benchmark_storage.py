"""
benchmark_storage.py — Benchmark comparatif MinIO vs RustFS (stockage S3-compatible).

Contrairement au pattern .env "un seul backend actif a la fois" utilise par
l'API (STORAGE_BACKEND/MINIO_ENDPOINT), ce script se connecte DIRECTEMENT aux
deux backends en parallele, via leurs ports hote distincts (cf. docker-compose.yml :
MinIO expose 9000/9001, RustFS expose 9002/9003 sur l'hote). Comme les deux
parlent le protocole S3, le meme client Python `minio` fonctionne pour les deux
-- pas besoin de modifier .env ni de redemarrer l'API entre deux runs.

Mesure, pour chaque taille de fichier testee : upload (put_object), download
(get_object), listing (list_objects) -- latence en percentiles + debit MB/s,
sur N repetitions par taille et par operation.

Usage:
    python benchmark_storage.py
    python benchmark_storage.py --sizes 10KB,1MB,10MB --repeats 10
    python benchmark_storage.py --backend rustfs   # un seul backend, pas de comparaison
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

from minio import Minio
from minio.error import S3Error

RESULTS_DIR = Path("./benchmark_results")

# Ports HOTE (pas les ports internes docker-compose) : le script tourne sur la
# machine hote, pas dans un conteneur, donc il doit passer par les ports
# publies (`ports:` dans docker-compose.yml), pas par les noms de service
# internes (minio:9000, rustfs:9000) qui ne sont resolus qu'a l'interieur du
# reseau Docker.
BACKENDS = {
    "minio": {
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "bucket": "benchmark-storage",
    },
    "rustfs": {
        "endpoint": "localhost:9002",
        "access_key": "rustfsadmin",
        "secret_key": "rustfsadmin",
        "bucket": "benchmark-storage",
    },
}


def parse_size(size_str: str) -> int:
    """Convertit '10KB', '1MB', '2GB' (ou un nombre brut d'octets) en octets."""
    s = size_str.strip().upper()
    if s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    if s.endswith("MB"):
        return int(float(s[:-2]) * 1024 * 1024)
    if s.endswith("GB"):
        return int(float(s[:-2]) * 1024 * 1024 * 1024)
    return int(s)


def _percentile(data: list, p: float):
    """Percentile par interpolation lineaire, sans dependance a numpy."""
    if not data:
        return None
    data_sorted = sorted(data)
    if len(data_sorted) == 1:
        return round(data_sorted[0], 4)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return round(data_sorted[f], 4)
    return round(data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f), 4)


def get_client(backend_name: str) -> Minio:
    cfg = BACKENDS[backend_name]
    return Minio(cfg["endpoint"], access_key=cfg["access_key"], secret_key=cfg["secret_key"], secure=False)


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def time_operation(fn, repeats: int) -> dict:
    """Chronometre `fn` (sans argument) `repeats` fois, renvoie stats agregees."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "avg_seconds": round(sum(times) / len(times), 4),
        "p50_seconds": _percentile(times, 50),
        "p95_seconds": _percentile(times, 95),
        "min_seconds": round(min(times), 4),
        "max_seconds": round(max(times), 4),
    }


def benchmark_backend(backend_name: str, sizes: dict, repeats: int) -> dict:
    print(f"\n=== Backend : {backend_name} ({BACKENDS[backend_name]['endpoint']}) ===")
    client = get_client(backend_name)
    bucket = BACKENDS[backend_name]["bucket"]

    try:
        ensure_bucket(client, bucket)
    except Exception as e:
        print(f"  [!] Connexion/creation du bucket impossible pour '{backend_name}' : {e}")
        print(f"      Verifiez que le conteneur '{backend_name}' tourne et que le port est bien publie.")
        return {}

    results = {}
    for size_label, size_bytes in sizes.items():
        print(f"  Taille {size_label} ({size_bytes} octets), {repeats} repetitions...")
        payload = os.urandom(size_bytes)
        object_name = f"bench_{size_label}.bin"

        def do_upload(payload=payload, object_name=object_name):
            client.put_object(bucket, object_name, io.BytesIO(payload), length=len(payload))

        def do_download(object_name=object_name):
            resp = client.get_object(bucket, object_name)
            try:
                resp.read()
            finally:
                resp.close()
                resp.release_conn()

        def do_list():
            list(client.list_objects(bucket))

        try:
            upload_stats = time_operation(do_upload, repeats)
            download_stats = time_operation(do_download, repeats)
            list_stats = time_operation(do_list, repeats)
        except S3Error as e:
            print(f"    [!] Erreur S3 sur {size_label} : {e}")
            continue

        mb = size_bytes / (1024 * 1024)
        upload_stats["throughput_mb_s"] = round(mb / upload_stats["avg_seconds"], 2) if upload_stats["avg_seconds"] > 0 else None
        download_stats["throughput_mb_s"] = round(mb / download_stats["avg_seconds"], 2) if download_stats["avg_seconds"] > 0 else None

        try:
            client.remove_object(bucket, object_name)
        except S3Error:
            pass

        results[size_label] = {
            "size_bytes": size_bytes,
            "upload": upload_stats,
            "download": download_stats,
            "list": list_stats,
        }

        print(f"    upload   : avg={upload_stats['avg_seconds']}s  p95={upload_stats['p95_seconds']}s  ({upload_stats['throughput_mb_s']} MB/s)")
        print(f"    download : avg={download_stats['avg_seconds']}s  p95={download_stats['p95_seconds']}s  ({download_stats['throughput_mb_s']} MB/s)")
        print(f"    list     : avg={list_stats['avg_seconds']}s")

    return results


def print_comparison(results_minio: dict, results_rustfs: dict, sizes: dict) -> None:
    print("\n" + "=" * 88)
    print("COMPARAISON MinIO vs RustFS")
    print("=" * 88)
    header = f"{'Taille':<8} | {'Operation':<9} | {'MinIO avg (s)':>13} | {'RustFS avg (s)':>14} | {'Gagnant':>10} | {'Ecart':>8}"
    print(header)
    print("-" * len(header))
    for size_label in sizes:
        for op in ("upload", "download"):
            a = results_minio.get(size_label, {}).get(op, {})
            b = results_rustfs.get(size_label, {}).get(op, {})
            avg_a, avg_b = a.get("avg_seconds"), b.get("avg_seconds")
            if avg_a is None or avg_b is None:
                continue
            if avg_a < avg_b:
                winner, ecart = "MinIO", f"{round(avg_b / avg_a, 2)}x"
            elif avg_b < avg_a:
                winner, ecart = "RustFS", f"{round(avg_a / avg_b, 2)}x"
            else:
                winner, ecart = "egalite", "1x"
            print(f"{size_label:<8} | {op:<9} | {avg_a:>13} | {avg_b:>14} | {winner:>10} | {ecart:>8}")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MinIO vs RustFS (stockage S3-compatible).")
    parser.add_argument("--sizes", type=str, default="10KB,100KB,1MB,10MB", help="Tailles a tester, separees par des virgules (ex: 10KB,1MB,10MB).")
    parser.add_argument("--repeats", type=int, default=5, help="Nombre de repetitions par taille et par operation.")
    parser.add_argument("--backend", choices=["minio", "rustfs", "both"], default="both")
    parser.add_argument(
        "--minio-endpoint", type=str, default=None,
        help="Endpoint MinIO (host:port). Defaut: localhost:9000 (execution sur l'hote). "
             "Utiliser 'minio:9000' pour une execution DANS un conteneur du meme reseau docker-compose.",
    )
    parser.add_argument(
        "--rustfs-endpoint", type=str, default=None,
        help="Endpoint RustFS (host:port). Defaut: localhost:9002 (execution sur l'hote, port publie). "
             "Utiliser 'rustfs:9000' pour une execution DANS un conteneur (port INTERNE, pas le port publie 9002).",
    )
    args = parser.parse_args()

    if args.minio_endpoint:
        BACKENDS["minio"]["endpoint"] = args.minio_endpoint
    if args.rustfs_endpoint:
        BACKENDS["rustfs"]["endpoint"] = args.rustfs_endpoint

    sizes = {s.strip(): parse_size(s) for s in args.sizes.split(",")}

    all_results = {}
    if args.backend in ("minio", "both"):
        all_results["minio"] = benchmark_backend("minio", sizes, args.repeats)
    if args.backend in ("rustfs", "both"):
        all_results["rustfs"] = benchmark_backend("rustfs", sizes, args.repeats)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "storage_benchmark.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultats sauvegardes dans : {output_path}")

    if "minio" in all_results and "rustfs" in all_results and all_results["minio"] and all_results["rustfs"]:
        print_comparison(all_results["minio"], all_results["rustfs"], sizes)


if __name__ == "__main__":
    main()