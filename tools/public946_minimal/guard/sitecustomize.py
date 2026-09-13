import os
if os.environ.get('PUBLIC946_INFERENCE') == '1':
    from public946_minimal.isolation import install
    install()
