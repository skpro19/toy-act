"""Select SSH endpoints from validated instance data, never CLI URL caches.

Vast's upstream vast.py::_ssh_url uses ports['22/tcp'][0]['HostPort']
with public_ipaddr for direct SSH, and ssh_host/ssh_port for the proxy.
https://github.com/vast-ai/vast-python/blob/master/vast.py
"""

from collections.abc import Callable
import ipaddress
from pathlib import Path
import re
import time

from workflow_common import Blocked, Journal, atomic_write


def endpoint(*, host: object, port: object) -> dict:
    if not isinstance(host, str) or not re.fullmatch(r"[a-zA-Z0-9.-]+", host) or host.startswith("-"):
        raise Blocked("Invalid provider SSH host")
    if isinstance(port, str) and port.isascii() and port.isdecimal():
        port = int(port)
    if type(port) is not int or not 1 <= port <= 65535:
        raise Blocked("Invalid provider SSH port")
    return {"host": host, "port": port}


def candidates(record: dict) -> list[dict]:
    result = []
    ports = record.get("ports", {})
    if not isinstance(ports, dict):
        raise Blocked("Invalid provider port mappings")
    mapping = ports.get("22/tcp")
    if mapping is not None:
        if not isinstance(mapping, list) or not mapping or any(not isinstance(binding, dict) for binding in mapping):
            raise Blocked("Ambiguous or invalid direct SSH mapping")
        try:
            address = ipaddress.IPv4Address(record.get("public_ipaddr"))
        except ipaddress.AddressValueError as error:
            raise Blocked("Invalid provider direct SSH address") from error
        if not address.is_global:
            raise Blocked("Direct SSH address must be public")
        # Docker can publish the same port for both IPv4 and IPv6. Validate
        # every binding, then accept duplicates only if their ports agree.
        direct = [endpoint(host=str(address), port=binding.get("HostPort")) for binding in mapping]
        if any(candidate != direct[0] for candidate in direct[1:]):
            raise Blocked("Ambiguous or invalid direct SSH mapping")
        result.append(direct[0])
    if record.get("ssh_host") is not None or record.get("ssh_port") is not None:
        proxy = endpoint(host=record.get("ssh_host"), port=record.get("ssh_port"))
        if "jupyter" in str(record.get("image_runtype", "")):
            proxy = endpoint(host=proxy["host"], port=proxy["port"] + 1)
        if proxy not in result:
            result.append(proxy)
    if not result:
        raise Blocked("Provider supplied no usable SSH endpoint")
    return result


def select_endpoint(*, journal: Journal, record: dict, attempt: dict,
                    known_hosts: Path, save: Callable[[], None]) -> dict:
    options = candidates(record)
    saved = attempt.get("ssh_endpoint")
    if saved is not None:
        if saved not in options:
            raise Blocked("Saved SSH endpoint changed; refusing automatic replacement")
        options = [saved]
    for _ in range(12):
        for candidate in options:
            host, port = candidate["host"], candidate["port"]
            result = journal.run(args=["ssh-keyscan", "-T", "10", "-p", str(port), host],
                                 timeout=35, check=False)
            lines = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
            if result.returncode or not lines:
                continue
            expected_host = host if port == 22 else f"[{host}]:{port}"
            if any(len(line.split()) != 3 or line.split()[0] != expected_host
                   or line.split()[1] not in {"ssh-rsa", "ssh-ed25519", "ecdsa-sha2-nistp256",
                                              "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521"} for line in lines):
                raise Blocked("Invalid SSH host-key scan output")
            keys = "\n".join(lines) + "\n"
            if known_hosts.exists() and set(known_hosts.read_text().splitlines()) != set(lines):
                raise Blocked("SSH host key changed; refusing automatic replacement")
            # Record the endpoint before writing keys or attempting remote setup.
            attempt["ssh_endpoint"] = candidate
            save()
            atomic_write(path=known_hosts, text=keys)
            return candidate
        time.sleep(5)
    raise Blocked("SSH host-key scans failed for provider endpoints; resume this iteration, not another rental")
