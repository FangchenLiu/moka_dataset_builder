import glob
import os
import struct
import tensorflow as tf
import tqdm

from typing import Optional
from tensorflow_datasets.core.writer import Writer
from tensorflow_datasets.core.shuffle import Shuffler, HKEY_SIZE_BYTES, _read_hkey


class ResumeWriter(Writer):
  """Shuffles and writes Examples to sharded TFRecord files.

  The number of shards is computed automatically.
  Can resume from an incomplete run.
  """

  def __init__(
      self,
      *args,
      resume_dir: Optional[str] = None,
      **kwargs,
  ):
    """Initializes Writer.

    Args:
      See Writer.
      resume_dir: Directory of incomplete files from previous run.
    """
    super().__init__(*args, **kwargs)

    # load sequences from previous run & add them to shuffler

    def read_bucket(file_path):
        with tf.io.gfile.GFile(file_path, 'rb') as fobj:
            while True:
                buff = fobj.read(HKEY_SIZE_BYTES)
                if not buff:
                    break
                hkey = _read_hkey(buff)
                size_bytes = fobj.read(8)
                size = struct.unpack('=Q', size_bytes)[0]
                data = fobj.read(size)
                yield hkey, data

    tmp_bucket_files = glob.glob(os.path.join(resume_dir, "bucket_*.tmp"))
    print(f"Found {len(tmp_bucket_files)} temp buckets from previous run.")
    if len(tmp_bucket_files) > 0:
        print("Resuming...")
        for tmp_bucket_file in tqdm.tqdm(tmp_bucket_files):
            for key, data in read_bucket(tmp_bucket_file):
                self._shuffler.add(key, data)
                self._num_examples += 1
        print("Finished resuming!")


class SafeShuffler(Shuffler):
    """
    Does not delete temp bucket files until all are iterated through
    (to prevent data loss if writer.finalize() fails)

    Caution: this will temporarily double the memory usage!
    """

    def _iter_buckets(self):
        for bucket in self._buckets:
            bucket_data = sorted(bucket.read_values())
            for hkey, data in bucket_data:
                yield hkey, data
        for bucket in self._buckets:
            bucket.del_file()