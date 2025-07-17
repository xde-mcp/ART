"""Utilities for working with S3 checkpoints."""

import asyncio
from asyncio.subprocess import PIPE


async def get_latest_checkpoint_step_from_s3(
    model_name: str,
    project: str,
    s3_bucket: str | None = None,
    prefix: str | None = None,
) -> int | None:
    """
    Get the latest checkpoint step number from S3 without downloading files.

    Returns:
        The latest step number, or None if no checkpoints exist.
    """
    from .s3 import build_s3_path

    s3_path = build_s3_path(
        model_name=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )

    # List checkpoint directories in S3
    cmd = ["aws", "s3", "ls", f"{s3_path}/checkpoints/"]

    process = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        # No checkpoints found or error
        return None

    # Parse output to find checkpoint directories
    lines = stdout.decode().strip().split("\n")
    checkpoint_steps = []

    for line in lines:
        if line.strip():
            # AWS S3 ls output format: "PRE 0001/"
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "PRE":
                dirname = parts[1].rstrip("/")
                if dirname.isdigit():
                    checkpoint_steps.append(int(dirname))

    return max(checkpoint_steps) if checkpoint_steps else None


async def get_checkpoint_step_not_after_from_s3(
    model_name: str,
    project: str,
    not_after_step: int,
    s3_bucket: str | None = None,
    prefix: str | None = None,
) -> int | None:
    """
    Get the latest checkpoint step number that is not after the specified step from S3.

    Args:
        not_after_step: Find the latest checkpoint <= this step.

    Returns:
        The step number, or None if no suitable checkpoint exists.
    """
    from .s3 import build_s3_path

    s3_path = build_s3_path(
        model_name=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )

    # List checkpoint directories in S3
    cmd = ["aws", "s3", "ls", f"{s3_path}/checkpoints/"]

    process = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        # No checkpoints found or error
        return None

    # Parse output to find checkpoint directories
    lines = stdout.decode().strip().split("\n")
    valid_steps = []

    for line in lines:
        if line.strip():
            # AWS S3 ls output format: "PRE 0001/"
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "PRE":
                dirname = parts[1].rstrip("/")
                if dirname.isdigit():
                    step = int(dirname)
                    if step <= not_after_step:
                        valid_steps.append(step)

    return max(valid_steps) if valid_steps else None
