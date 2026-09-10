"""One-way Python network denial for optional local provider execution."""
import ipaddress
import os
import sys

_installed=False

def deny_external_network():
    global _installed
    if _installed:return
    def audit(event,args):
        if event in ['socket.getaddrinfo','socket.gethostbyname']:
            host=args[0]
        elif event in ['socket.connect','socket.sendto'] and isinstance(args[1],tuple):
            host=args[1][0]
        else:return
        if host in ['',None,'localhost',os.uname().nodename]:return
        try:local=ipaddress.ip_address(host).is_loopback
        except ValueError:local=False
        if not local:raise PermissionError('v6 local providers cannot download or call external services')
    sys.addaudithook(audit);_installed=True
