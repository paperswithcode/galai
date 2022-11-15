import os
import re
import tqdm
import urllib

from galai.consts import CHECKPOINT_PATHS, TOKENIZER_URL

# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"

def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

def _get_cache_home():
    cache_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                         DEFAULT_CACHE_DIR), 'galactica')))
    return cache_home


class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_file(file_url: str, file_loc: str):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=file_url.split('/')[-1]) as t:
        urllib.request.urlretrieve(file_url, filename=file_loc, reporthook=t.update_to)

def download_model(model_name: str, model_path: str):

    for file_url in tqdm.tqdm(CHECKPOINT_PATHS[model_name]):
        file_loc = os.path.join(model_path, file_url.split('/')[-1])
        if os.path.exists(file_loc):
            continue
        _download_file(file_url, file_loc)

def download_tokenizer(tokenizer_path: str):
    _download_file(TOKENIZER_URL, tokenizer_path)

def get_checkpoint_path(model_name: str) -> str:
    """
    Downloads checkpoint if not in the ~/.cache/galai/ directory.
    Once all files are available, it returns the path.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'mini'

    Returns
    ----------
    str - the path of the model weights
    """
    cache_dir = _get_cache_home()

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    model_path = os.path.join(cache_dir, f"{model_name}.pt")

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if os.path.exists(model_path):
        for file in CHECKPOINT_PATHS[model_name]:
            file_name = os.path.join(model_path, file.split('/')[-1])
            if not os.path.exists(file_name):
                print('Incomplete files for model; downloading')
                download_model(model_name=model_name, model_path=model_path)
        return model_path
    else:
        download_model(model_name=model_name, model_path=model_path)
        return model_path

def get_tokenizer_path() -> str:
    """
    Downloads tokenizer if not in the ~/.cache/galai/ directory.
    Once all files are available, it returns the path.

    Returns
    ----------
    str - the path of the tokenizer
    """
    cache_dir = _get_cache_home()

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    tokenizer_path = os.path.join(cache_dir, 'tokenizer')

    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)

    file_name = os.path.join(tokenizer_path, 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        if not os.path.exists(file_name):
            print('Incomplete files for tokenizer; downloading')
            download_tokenizer(file_name)
        return file_name
    else:
        download_tokenizer(file_name)
        return file_name
