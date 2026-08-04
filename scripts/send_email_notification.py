"""Send one short notification email over SMTP.

Used by `scripts/notify_experiments.sh` as its email backend, but it works on
its own for anything that wants to mail a line of text at the end of a long job:

    SMTP_USER=you@gmail.com SMTP_PASS=... NOTIFY_EMAIL_TO=you@gmail.com \\
        uv run python -m scripts.send_email_notification "Subject" "Body"

Configuration is read from the environment so the password never appears in a
command line or in shell history:

    NOTIFY_EMAIL_TO    recipient; comma-separated for several  (required)
    SMTP_USER          SMTP username                           (required)
    SMTP_PASS          SMTP password or app password           (required)
    NOTIFY_EMAIL_FROM  sender address                          (default SMTP_USER)
    SMTP_HOST          server                                  (default smtp.gmail.com)
    SMTP_PORT          port                                    (default 465)
    SMTP_SSL           1 for implicit TLS, 0 for STARTTLS      (default 1)

For Gmail, SMTP_PASS must be an App Password, which requires 2-Step
Verification on the account; a normal account password is rejected.

Exit codes: 0 sent, 1 misconfigured, 2 sending failed.
"""

from __future__ import annotations

import os
import smtplib
import ssl
import sys
from email.message import EmailMessage


def build_message(subject: str, body: str, sender: str, recipients: list[str]) -> EmailMessage:
    """Build a plain-text email.

    Args:
        subject: Subject line.
        body: Message body.
        sender: From address.
        recipients: One or more recipient addresses.

    Returns:
        The assembled message.
    """
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(body)
    return message


def send(message: EmailMessage, host: str, port: int, user: str, password: str, use_ssl: bool):
    """Deliver a message, using implicit TLS or STARTTLS.

    Authentication is skipped when no password is set, which is what lets this
    talk to a local debug SMTP server during testing.

    Args:
        message: The message to send.
        host: SMTP server hostname.
        port: SMTP server port.
        user: SMTP username.
        password: SMTP password; empty to skip authentication.
        use_ssl: True for implicit TLS, False for STARTTLS.
    """
    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
            if password:
                server.login(user, password)
            server.send_message(message)
        return

    with smtplib.SMTP(host, port, timeout=30) as server:
        if password:
            server.starttls(context=ssl.create_default_context())
            server.login(user, password)
        server.send_message(message)


def main(argv: list[str]) -> int:
    """Read configuration from the environment and send one email."""
    if len(argv) < 3:
        print("usage: send_email_notification.py <subject> <body>", file=sys.stderr)
        return 1

    subject, body = argv[1], argv[2]
    recipients = [r.strip() for r in os.environ.get("NOTIFY_EMAIL_TO", "").split(",") if r.strip()]
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASS", "")
    sender = os.environ.get("NOTIFY_EMAIL_FROM") or user

    if not recipients or not sender:
        print("NOTIFY_EMAIL_TO and SMTP_USER (or NOTIFY_EMAIL_FROM) are required", file=sys.stderr)
        return 1

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    use_ssl = os.environ.get("SMTP_SSL", "1") == "1"

    try:
        send(build_message(subject, body, sender, recipients), host, port, user, password, use_ssl)
    except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
        print(f"email delivery failed: {exc}", file=sys.stderr)
        return 2

    print(f"emailed {', '.join(recipients)}: {subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
