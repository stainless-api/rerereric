import tempfile, os, sys
import unittest.mock as mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rerereric.core import Rerereric

def make_rerereric():
    with mock.patch.object(Rerereric, '_get_git_dir', return_value='/tmp'):
        return Rerereric()

def test_crlf_preserved():
    r = make_rerereric()
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
        f.write(b'line1\r\nline2\r\nline3\r\n')
        path = f.name

    try:
        r._apply_resolution(path, {'start_line': 1, 'end_line': 1}, 'replaced')
        with open(path, 'rb') as f:
            result = f.read()
        assert result == b'line1\r\nreplaced\r\nline3\r\n', f'Unexpected: {result}'
        print('PASS: CRLF file stays CRLF')
    finally:
        os.unlink(path)

def test_lf_preserved():
    r = make_rerereric()
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
        f.write(b'line1\nline2\nline3\n')
        path = f.name

    try:
        r._apply_resolution(path, {'start_line': 1, 'end_line': 1}, 'replaced')
        with open(path, 'rb') as f:
            result = f.read()
        assert result == b'line1\nreplaced\nline3\n', f'Unexpected: {result}'
        print('PASS: LF file stays LF')
    finally:
        os.unlink(path)

if __name__ == '__main__':
    test_crlf_preserved()
    test_lf_preserved()
