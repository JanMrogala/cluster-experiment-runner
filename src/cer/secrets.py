"""CLI for managing CER secrets in the system keyring."""

import sys

import keyring

from cer.config import KEYRING_SERVICE, SECRET_FIELDS

VALID_KEYS = {f.split(".", 1)[1] for f in SECRET_FIELDS}


def main():
    args = sys.argv[1:]

    if len(args) < 1 or args[0] in ("-h", "--help"):
        _usage()
        return

    cmd = args[0]

    if cmd == "set" and len(args) == 3:
        key, value = args[1], args[2]
        _validate_key(key)
        keyring.set_password(KEYRING_SERVICE, key, value)
        print(f"Stored {key} in system keyring.")

    elif cmd == "get" and len(args) == 2:
        key = args[1]
        _validate_key(key)
        value = keyring.get_password(KEYRING_SERVICE, key)
        if value is None:
            print(f"{key} not found in system keyring.", file=sys.stderr)
            sys.exit(1)
        print(value)

    elif cmd == "delete" and len(args) == 2:
        key = args[1]
        _validate_key(key)
        try:
            keyring.delete_password(KEYRING_SERVICE, key)
            print(f"Deleted {key} from system keyring.")
        except keyring.errors.PasswordDeleteError:
            print(f"{key} not found in system keyring.", file=sys.stderr)
            sys.exit(1)

    else:
        _usage()
        sys.exit(1)


def _validate_key(key: str):
    if key not in VALID_KEYS:
        print(f"Unknown key: {key}. Valid keys: {', '.join(sorted(VALID_KEYS))}", file=sys.stderr)
        sys.exit(1)


def _usage():
    keys = ", ".join(sorted(VALID_KEYS))
    print(f"Usage: cer-secret <set|get|delete> <key> [value]")
    print(f"Valid keys: {keys}")
    print()
    print("Examples:")
    print("  cer-secret set wandb_api_key <your-key>")
    print("  cer-secret get wandb_api_key")
    print("  cer-secret delete wandb_api_key")


if __name__ == "__main__":
    main()
