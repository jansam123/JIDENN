from dataclasses import dataclass

@dataclass
class Dataset:
    batch_size: int   # Batch size.
    validation_step: int   # Validation every n batches.
    reading_size: int   # Number of events to load at a time.
    num_workers: int   # Number of workers to use when loading data.
    take: int | None   # Length of data to use.
    validation_batches: int   # Size of validation dataset.
    dev_size: float    # Size of dev dataset.
    test_size: float   # Size of test dataset.
    shuffle_buffer: int | None   # Size of shuffler buffer.