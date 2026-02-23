---
name: cloud-vm-sync
description: Use when deploying, syncing, or copying Python and JSON files to the live STAVKI virtual machine, or when restarting the Telegram bot service.
---

# Cloud VM Sync Protocol

## Overview

Remote deployments are highly sensitive. Blindly pushing untested scripts or skipping daemon restarts creates silent downtime on the Virtual Machine.

**Core principle:** ALWAYS run local syntax evaluation before pushing. ALWAYS read the remote syslog tail after a daemon restart.

## The Iron Law

```
NO CODE MAY BE RSYNC'D TO THE LIVE VM UNLESS IT HAS PASSED LOCAL COMPILATION FIRST.
```

## When to Use

Use this specific protocol when:
- Executing updates to the `stavki_bot.service`.
- Changing config `.json` values (`kelly_state`, `synonyms`).
- Pushing core Python scripts like `kelly.py` or feature pipelines.

**Use this ESPECIALLY when:**
- The user declares an emergency live hotfix.
- You are updating a single line of Python syntax.

**Don't skip when:**
- It is "just a config file."
- You are in a rush to prove a fix works.

## The Sync Phases

### Phase 1: Local Diagnostics (Pre-flight)

**BEFORE executing any `scp` or `rsync` commands:**

1. **Verify Python Syntax**
   - Syntax errors crash the STAVKI systemd process instantly.
   ```bash
   # ❌ Bad: Pushing python blindly
   rsync -avz bot.py user@vm:~/
   
   # ✅ Good: Testing compilation first
   python -m py_compile stavki/interfaces/telegram_bot.py
   ```

2. **Verify JSON Configurations**
   - Misplaced commas in state dicts permanently break the STAVKI parsing loop.
   ```bash
   # ✅ Good: Validate JSON structure locally
   python -m json.tool config/users/nick_kelly_state.json > /dev/null
   ```

### Phase 2: Secure Transfer

**Execute the transfer using `rsync` with the strict GCP SSH configurations:**

1. **Staging the File**
   Always sync the file into the `macuser` home directory first, NEVER into the root pipeline.
   ```bash
   rsync -avz -e "ssh -i /Users/macuser/.ssh/google_compute_engine -o StrictHostKeyChecking=no" <local_file> serni13678@34.185.182.219:~/stavki_v2/
   ```

2. **Injecting the File**
   Apply `sudo` to push the file into the operational system and enforce correct STAVKI permissions:
   ```bash
   ssh -i /Users/macuser/.ssh/google_compute_engine -o StrictHostKeyChecking=no serni13678@34.185.182.219 "sudo cp ~/stavki_v2/<file> /home/macuser/stavki_v2/<file> && sudo chown macuser:macuser /home/macuser/stavki_v2/<file>"
   ```

### Phase 3: Runtime Verification

**You must force the background bot to digest the new file.**

1. **Restart the Bot Daemon**
   ```bash
   ssh -i /Users/macuser/.ssh/google_compute_engine -o StrictHostKeyChecking=no serni13678@34.185.182.219 "sudo systemctl restart stavki_bot.service && sleep 2 && systemctl status stavki_bot.service --no-pager"
   ```

2. **Diagnose Crashes**
   If the `systemctl status` command returns failure:
   ```bash
   # Fetch the exact Python traceback causing the crash
   journalctl -u stavki_bot.service -n 50 --no-pager
   ```

## Red Flags - STOP and Follow Process

If you catch yourself thinking:
- "The syntax is so simple, I'll just push it directly."
- "I'll restart the service later after pushing three more files."
- "The status command failed, I'll just try restarting it again without looking at `journalctl`."
- "The JSON edit was just changing `0.05` to `1.0`, no need to validate."

**ALL of these mean: STOP. You are causing silent downtime.**

## Quick Reference 

| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Diagnostics** | `py_compile`, JSON checks | No syntax errors thrown. |
| **2. Transfer** | `rsync`, `chown` | File lands with correct permissions. |
| **3. Verification** | `systemctl restart`, `status` | Bot returns `active (running)`. |
