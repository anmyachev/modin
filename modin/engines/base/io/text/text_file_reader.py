# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from modin.engines.base.io.file_reader import FileReader
import numpy as np
import warnings
import os
import io


class TextFileReader(FileReader):
    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        return np.array(
            [
                [
                    cls.frame_partition_cls(
                        partition_ids[i][j],
                        length=row_lengths[i],
                        width=column_widths[j],
                    )
                    for j in range(len(partition_ids[i]))
                ]
                for i in range(len(partition_ids))
            ]
        )

    @classmethod
    def pathlib_or_pypath(cls, filepath_or_buffer):
        try:
            import py

            if isinstance(filepath_or_buffer, py.path.local):
                return True
        except ImportError:  # pragma: no cover
            pass
        try:
            import pathlib

            if isinstance(filepath_or_buffer, pathlib.Path):
                return True
        except ImportError:  # pragma: no cover
            pass
        return False

    @classmethod
    def offset(
        cls,
        f,
        offset_size: int,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
        encoding: str = None,
        newline: bytes = None,
    ):
        """
        Moves the file offset at the specified amount of bytes.

        Parameters
        ----------
        f: file object
        offset_size: int
            Number of bytes to read and ignore.
        quotechar: bytes, default b'"'
            Indicate quote in a file.
        is_quoting: bool, default True
            Whether or not to consider quotes.
        encoding: str, optional
            Encoding of `f`.
        newline: bytes, optional
            Byte or sequence of bytes indicating line endings.

        Returns
        -------
        bool
            If file pointer reached the end of the file, but did not find
            closing quote returns `False`. `True` in any other case.
        """
        if is_quoting:
            chunk = f.read(offset_size)
            outside_quotes = (
                not (chunk.count(quotechar) - chunk.count(b"\\" + quotechar)) % 2
            )
        else:
            f.seek(offset_size, os.SEEK_CUR)
            outside_quotes = True

        # after we read `offset_size` bytes, we most likely break the line but
        # the modin implementation doesn't work correctly in the case, so we must
        # make sure that the line is read completely to the lineterminator,
        # which is what the `_read_rows` does
        outside_quotes, _ = cls._read_rows(
            f,
            nrows=1,
            quotechar=quotechar,
            is_quoting=is_quoting,
            outside_quotes=outside_quotes,
            encoding=encoding,
            newline=newline,
        )

        return outside_quotes

    @classmethod
    def partitioned_file(
        cls,
        f,
        num_partitions: int = None,
        nrows: int = None,
        skiprows: int = None,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
        encoding: str = None,
        newline: bytes = None,
    ):
        """
        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        f: file to be partitioned
        num_partitions: int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.pandas.DEFAULT_NPARTITIONS`
        nrows: int, optional
            Number of rows of file to read.
        skiprows: array or callable, optional
            Specifies rows to skip.
        quotechar: bytes, default b'"'
            Indicate quote in a file.
        is_quoting: bool, default True
            Whether or not to consider quotes.
        encoding: str, optional
            Encoding of `f`.
        newline: bytes, optional
            Byte or sequence of bytes indicating line endings.

        Returns
        -------
        An array, where each element of array is a tuple of two ints:
        beginning and the end offsets of the current chunk.
        """
        if num_partitions is None:
            from modin.pandas import DEFAULT_NPARTITIONS

            num_partitions = DEFAULT_NPARTITIONS

        result = []
        file_size = cls.file_size(f)

        if skiprows:
            outside_quotes, read_rows = cls._read_rows(
                f,
                nrows=skiprows,
                quotechar=quotechar,
                is_quoting=is_quoting,
                encoding=encoding,
                newline=newline,
            )

        start = f.tell()

        if nrows:
            read_rows_counter = 0
            partition_size = max(1, num_partitions, nrows // num_partitions)
            while f.tell() < file_size and read_rows_counter < nrows:
                if read_rows_counter + partition_size > nrows:
                    # it's possible only if is_quoting==True
                    partition_size = nrows - read_rows_counter
                outside_quotes, read_rows = cls._read_rows(
                    f,
                    nrows=partition_size,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                    encoding=encoding,
                    newline=newline,
                )
                result.append((start, f.tell()))
                start = f.tell()
                read_rows_counter += read_rows

                # add outside_quotes
                if is_quoting and not outside_quotes:
                    warnings.warn("File has mismatched quotes")
        else:
            partition_size = max(1, num_partitions, file_size // num_partitions)
            while f.tell() < file_size:
                outside_quotes = cls.offset(
                    f,
                    offset_size=partition_size,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                    encoding=encoding,
                    newline=newline,
                )

                result.append((start, f.tell()))
                start = f.tell()

                # add outside_quotes
                if is_quoting and not outside_quotes:
                    warnings.warn("File has mismatched quotes")

        return result

    @classmethod
    def _read_rows(
        cls,
        f,
        nrows: int,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
        outside_quotes: bool = True,
        encoding: str = None,
        newline: bytes = None,
    ):
        """
        Move the file offset at the specified amount of rows.

        Parameters
        ----------
        f: file object
        nrows: int
            Number of rows to read.
        quotechar: bytes, default b'"'
            Indicate quote in a file.
        is_quoting: bool, default True
            Whether or not to consider quotes.
        outside_quotes: bool, default True
            Whether the file pointer is within quotes or not at the time this function is called.
        encoding: str, optional
            Encoding of `f`.
        newline: bytes, optional
            Byte or sequence of bytes indicating line endings.

        Returns
        -------
        tuple of bool and int,
            bool: If file pointer reached the end of the file, but did not find
            closing quote returns `False`. `True` in any other case.
            int: Number of rows that was read.
        """
        if nrows is not None and nrows <= 0:
            return True, 0

        rows_read = 0

        if encoding not in ("unicode_escape", "utf16", "utf32"):
            for line in f:
                if (
                    is_quoting
                    and (line.count(quotechar) - line.count(b"\\" + quotechar)) % 2
                ):
                    outside_quotes = not outside_quotes
                if outside_quotes:
                    rows_read += 1
                    if rows_read >= nrows:
                        break
        else:
            buffer_size = io.DEFAULT_BUFFER_SIZE
            chunk = f.read(buffer_size)
            while chunk:
                bytes_read = 0
                # split remove newline bytes from line
                lines = chunk.split(newline)
                chunk_size = len(chunk)
                if len(lines) != 1:
                    for line in lines[:-1]:
                        bytes_read += len(line) + len(newline)
                        if (
                            is_quoting
                            and (line.count(quotechar) - line.count(b"\\" + quotechar))
                            % 2
                        ):
                            outside_quotes = not outside_quotes
                        if outside_quotes:
                            rows_read += 1
                            if rows_read >= nrows:
                                f.seek(f.tell() + bytes_read - chunk_size)
                                return outside_quotes, rows_read

                chunk = f.read(buffer_size)
                if lines[-1]:
                    # last line can be read without newline bytes
                    chunk = lines[-1] + chunk

        # case when EOF
        if not outside_quotes:
            rows_read += 1

        return outside_quotes, rows_read

    @classmethod
    def compute_newline(cls, fpath_or_buf, encoding):
        newline = None
        if encoding in ("unicode_escape", "utf16", "utf32"):
            import codecs

            with open(fpath_or_buf, "r", encoding=encoding, newline="") as str_f:
                try:
                    # trigger for computing f.newlines
                    str_f.readline()
                except UnicodeDecodeError as e:
                    with open(fpath_or_buf, "rb") as f:
                        buffer = f.read(8200)
                    raise ValueError(buffer + b"\n\n" + e.args[0].encode("utf8"))
                # in bytes
                newline = str_f.newlines.encode(encoding)
                if encoding == "utf16":
                    if newline.startswith(codecs.BOM_UTF16):
                        newline = newline[len(codecs.BOM_UTF16) :]
                elif encoding == "utf32":
                    if newline.startswith(codecs.BOM_UTF32):
                        newline = newline[len(codecs.BOM_UTF32) :]
        return newline
