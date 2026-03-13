import hashlib

def sha256_hash(data: Union[str, bytes]) -> str:

    if isinstance(data, bytes):
        payload = data

    elif isinstance(data, str):
        payload = data.encode("utf-8")

    else:
        raise TypeError("sha256_hash expects str or bytes")

    return hashlib.sha256(payload).hexdigest()
