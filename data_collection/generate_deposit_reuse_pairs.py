from service.deposit_reuse_pairs_job import DepositReusePairJob


def generate_deposit_reuse_pairs(
    chain_name: str,
    file_path: str = "deposit_reuse_pairs.csv",
    max_workers: int = 2,
    batch_size: int = 1000,
):
    """Generate deposit reuse wallet pairs and save to file.

    Args:
        chain_name (str): Blockchain name (e.g., 'bsc', 'ethereum').
        file_path (str): Output CSV path to save pairs.
        max_workers (int): Number of parallel workers.
        batch_size (int): Batch size for processing.
    """
    job = DepositReusePairJob(
        chain=chain_name,
        path=file_path,
        max_workers=max_workers,
        batch_size=batch_size,
    )
    job.run()
