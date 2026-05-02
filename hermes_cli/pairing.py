"""
CLI commands for the DM pairing system.

Usage:
    hermes pairing list              # Show all pending + approved users
    hermes pairing approve <platform> <code>  # Approve a pairing code
    hermes pairing revoke <platform> <user_id> # Revoke user access
    hermes pairing clear-pending     # Clear all expired/pending codes
    hermes pairing menu              # Interactive curses menu
"""
import time


def pairing_command(args):
    """Handle hermes pairing subcommands."""
    from gateway.pairing import PairingStore

    store = PairingStore()
    action = getattr(args, "pairing_action", None)

    if action == "list":
        _cmd_list(store)
    elif action == "approve":
        _cmd_approve(store, args.platform, args.code)
    elif action == "revoke":
        _cmd_revoke(store, args.platform, args.user_id)
    elif action == "clear-pending":
        _cmd_clear_pending(store)
    elif action == "menu":
        _cmd_menu(store)
    else:
        print("Usage: hermes pairing {list|approve|revoke|clear-pending|menu}")
        print("Run 'hermes pairing --help' for details.")


def _cmd_list(store):
    """List all pending and approved users."""
    pending = store.list_pending()
    approved = store.list_approved()

    if not pending and not approved:
        print("No pairing data found. No one has tried to pair yet~")
        return

    if pending:
        print(f"\n  Pending Pairing Requests ({len(pending)}):")
        print(f"  {'Platform':<12} {'Code':<10} {'User ID':<20} {'Name':<20} {'Age'}")
        print(f"  {'--------':<12} {'----':<10} {'-------':<20} {'----':<20} {'---'}")
        for p in pending:
            print(
                f"  {p['platform']:<12} {p['code']:<10} {p['user_id']:<20} "
                f"{(p.get('user_name') or ''):<20} {p['age_minutes']}m ago"
            )
    else:
        print("\n  No pending pairing requests.")

    if approved:
        print(f"\n  Approved Users ({len(approved)}):")
        print(f"  {'Platform':<12} {'User ID':<20} {'Name':<20}")
        print(f"  {'--------':<12} {'-------':<20} {'----':<20}")
        for a in approved:
            print(f"  {a['platform']:<12} {a['user_id']:<20} {(a.get('user_name') or ''):<20}")
    else:
        print("\n  No approved users.")

    print()


def _cmd_approve(store, platform: str, code: str):
    """Approve a pairing code."""
    platform = platform.lower().strip()
    code = code.upper().strip()

    result = store.approve_code(platform, code)
    if result:
        uid = result["user_id"]
        name = result.get("user_name") or ""
        display = f"{name} ({uid})" if name else uid
        print(f"\n  Approved! User {display} on {platform} can now use the bot~")
        print("  They'll be recognized automatically on their next message.\n")
    else:
        print(f"\n  Code '{code}' not found or expired for platform '{platform}'.")
        print("  Run 'hermes pairing list' to see pending codes.\n")


def _cmd_revoke(store, platform: str, user_id: str):
    """Revoke a user's access."""
    platform = platform.lower().strip()

    if store.revoke(platform, user_id):
        print(f"\n  Revoked access for user {user_id} on {platform}.\n")
    else:
        print(f"\n  User {user_id} not found in approved list for {platform}.\n")


def _cmd_clear_pending(store):
    """Clear all pending pairing codes."""
    count = store.clear_pending()
    if count:
        print(f"\n  Cleared {count} pending pairing request(s).\n")
    else:
        print("\n  No pending requests to clear.\n")


def _cmd_menu(store):
    """Interactive curses menu — pick a section, then revoke/unblock users."""
    from hermes_cli.curses_ui import curses_single_select

    while True:
        approved = store.list_approved()
        limiter, blocked = _load_blocked()

        sections = [
            f"Approved users ({len(approved)})",
            f"Blocked users ({len(blocked)})",
        ]
        choice = curses_single_select(
            "  Pairing Manager",
            sections,
            default_index=0,
            cancel_label="Quit",
        )
        if choice is None:
            print()
            return
        if choice == 0:
            _menu_approved(store, approved)
        elif choice == 1:
            _menu_blocked(limiter, blocked)


def _load_blocked():
    """Best-effort load of the telegram blocked-users list."""
    try:
        from gateway.owner_approval import OwnerApprovalRateLimiter
        limiter = OwnerApprovalRateLimiter(platform="telegram")
        return limiter, limiter.list_blocked()
    except Exception:
        return None, []


def _menu_approved(store, approved):
    """Multi-select revoke flow for approved users."""
    if not approved:
        print("\n  No approved users.\n")
        _wait_enter()
        return
    items = [_fmt_approved(a) for a in approved]
    chosen = _multi_select(
        "  Approved Users — SPACE select, ENTER to revoke", items
    )
    if not chosen or not _confirm(f"Revoke {len(chosen)} user(s)?"):
        return
    revoked = 0
    for idx in sorted(chosen):
        a = approved[idx]
        if store.revoke(a["platform"], a["user_id"]):
            revoked += 1
    print(f"\n  Revoked {revoked} of {len(chosen)} user(s).\n")
    _wait_enter()


def _menu_blocked(limiter, blocked):
    """Multi-select unblock flow for blocked users."""
    if limiter is None or not blocked:
        print("\n  No blocked users.\n")
        _wait_enter()
        return
    items = [_fmt_blocked(b) for b in blocked]
    chosen = _multi_select(
        "  Blocked Users — SPACE select, ENTER to unblock", items
    )
    if not chosen or not _confirm(f"Unblock {len(chosen)} user(s)?"):
        return
    unblocked = 0
    for idx in sorted(chosen):
        b = blocked[idx]
        if limiter.unblock(b["user_id"]):
            unblocked += 1
    print(f"\n  Unblocked {unblocked} of {len(chosen)} user(s).\n")
    _wait_enter()


def _multi_select(title, items):
    """Wrap curses_checklist; return set of chosen indices (empty on cancel)."""
    from hermes_cli.curses_ui import curses_checklist
    return curses_checklist(title, items, set(), cancel_returns=set())


def _fmt_approved(a):
    age = _age_str(a.get("approved_at"))
    name = (a.get("user_name") or "—")[:20]
    return f"{a['platform']:<10} {a['user_id']:<14} {name:<20} approved {age}"


def _fmt_blocked(b):
    age = _age_str(b.get("blocked_at"))
    return f"{b['platform']:<10} {b['user_id']:<14} blocked {age}"


def _age_str(ts):
    if not ts:
        return "?"
    try:
        delta = max(0.0, time.time() - float(ts))
    except (TypeError, ValueError):
        return "?"
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _confirm(prompt):
    try:
        return input(f"\n  {prompt}  [y/N] ").strip().lower() == "y"
    except (KeyboardInterrupt, EOFError):
        return False


def _wait_enter():
    try:
        input("  Press Enter to continue... ")
    except (KeyboardInterrupt, EOFError):
        pass
