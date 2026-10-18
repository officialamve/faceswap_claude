# Security notes

* **Single administrator.** Every update is authorised by numeric Telegram user ID (`ADMIN_TELEGRAM_ID`), in a private chat
  whose chat ID equals that ID, *before* any media is downloaded. Usernames are never used.
* **Webhook** requests must carry Telegram's `X-Telegram-Bot-Api-Secret-Token` equal to `WEBHOOK_SECRET`
  (constant-time comparison); bodies are capped at 1 MB. There is no other public endpoint and no endpoint that
  performs a swap, fetches URLs, or touches arbitrary paths.
* **Secrets** live only in environment variables. Logs redact the bot token and webhook secret; Telegram API errors are
  sanitised so the token-bearing URL is never included.
* **Inputs** are validated by file signature *and* real decoding, with limits on bytes, pixels and dimensions
  (decompression-bomb guard). Temp files use generated names in a private (0700) directory; nothing derives paths from
  user-supplied names.
* **Data lifecycle.** Inputs and outputs live in a private temp directory only while needed and are deleted in `finally`
  blocks, on cancel/reset, at startup, and periodically. Nothing is stored persistently; no embeddings are kept. Deleting
  local files cannot delete copies already delivered to, or retained by, Telegram.
* **Misuse.** The bot is for photographs you have the right and consent to use. It has no features for intimate imagery or
  impersonation, and you remain responsible for lawful use.
* **Reporting.** Open a private security advisory on the GitHub repository.
