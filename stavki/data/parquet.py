"""
Parquet data format utilities.

Provides 5-10x faster loading compared to CSV.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


def csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
    compression: str = 'snappy'
) -> Path:
    """
    Convert CSV to Parquet format.
    
    Args:
        csv_path: Path to CSV file
        parquet_path: Output path (default: same name with .parquet)
        compression: Compression type ('snappy', 'gzip', 'brotli', None)
        
    Returns:
        Path to created parquet file
    """
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)
    
    logger.info(f"Converting {csv_path.name} to Parquet...")
    
    # Read CSV
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Convert to Parquet
    df.to_parquet(parquet_path, compression=compression, index=False)
    
    # Size comparison
    csv_size = csv_path.stat().st_size / 1024 / 1024
    parquet_size = parquet_path.stat().st_size / 1024 / 1024
    
    logger.info(f"  CSV: {csv_size:.1f} MB -> Parquet: {parquet_size:.1f} MB ({parquet_size/csv_size:.1%})")
    
    return parquet_path


def load_data(
    path: Union[str, Path],
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Smart data loader - uses Parquet if available, falls back to CSV.
    
    Args:
        path: Path to data file (with or without extension)
        columns: Optional list of columns to load (Parquet only)
        
    Returns:
        DataFrame
    """
    path = Path(path)
    
    # Try Parquet first
    parquet_path = path.with_suffix('.parquet')
    if parquet_path.exists():
        logger.debug(f"Loading Parquet: {parquet_path.name}")
        if columns:
            return pd.read_parquet(parquet_path, columns=columns)
        return pd.read_parquet(parquet_path)
    
    # Fall back to CSV
    csv_path = path.with_suffix('.csv')
    if csv_path.exists():
        logger.debug(f"Loading CSV: {csv_path.name}")
        if columns:
            return pd.read_csv(csv_path, usecols=columns, low_memory=False)
        return pd.read_csv(csv_path, low_memory=False)
    
    # Try exact path
    if path.exists():
        if path.suffix == '.parquet':
            return pd.read_parquet(path, columns=columns)
        return pd.read_csv(path, low_memory=False)
    
    raise FileNotFoundError(f"No data file found: {path}")


def convert_all_csv_to_parquet(data_dir: Union[str, Path]) -> List[Path]:
    """
    Convert all CSV files in directory to Parquet.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        List of created Parquet paths
    """
    data_dir = Path(data_dir)
    created = []
    
    for csv_file in data_dir.glob('*.csv'):
        parquet_path = csv_to_parquet(csv_file)
        created.append(parquet_path)
    
    logger.info(f"Converted {len(created)} files to Parquet")
    return created


def benchmark_loading(path: Union[str, Path], n_runs: int = 5) -> dict:
    """
    Benchmark CSV vs Parquet loading times.
    
    Args:
        path: Base path (without extension)
        n_runs: Number of runs for averaging
        
    Returns:
        Dict with timing results
    """
    import time
    
    path = Path(path)
    csv_path = path.with_suffix('.csv')
    parquet_path = path.with_suffix('.parquet')
    
    results = {'csv_times': [], 'parquet_times': []}
    
    # Benchmark CSV
    if csv_path.exists():
        for _ in range(n_runs):
            start = time.time()
            _ = pd.read_csv(csv_path, low_memory=False)
            results['csv_times'].append(time.time() - start)
    
    # Benchmark Parquet
    if parquet_path.exists():
        for _ in range(n_runs):
            start = time.time()
            _ = pd.read_parquet(parquet_path)
            results['parquet_times'].append(time.time() - start)
    
    results['csv_avg'] = sum(results['csv_times']) / len(results['csv_times']) if results['csv_times'] else 0
    results['parquet_avg'] = sum(results['parquet_times']) / len(results['parquet_times']) if results['parquet_times'] else 0
    results['speedup'] = results['csv_avg'] / results['parquet_avg'] if results['parquet_avg'] > 0 else 0
    
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/macuser/Documents/something/stavki_v2')
    
    from stavki.config import DATA_DIR
    
    print("Converting data to Parquet format...")
    print()
    
    # Convert main files
    for csv_name in ['features_full.csv', 'training_data.csv']:
        csv_path = DATA_DIR / csv_name
        if csv_path.exists():
            parquet_path = csv_to_parquet(csv_path)
            print(f"  ✅ {csv_name} -> {parquet_path.name}")
    
    print()
    print("Benchmarking load times...")
    results = benchmark_loading(DATA_DIR / 'features_full')
    print(f"  CSV:     {results['csv_avg']:.3f}s")
    print(f"  Parquet: {results['parquet_avg']:.3f}s")
    print(f"  Speedup: {results['speedup']:.1f}x")
