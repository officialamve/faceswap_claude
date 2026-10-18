import unittest

from app.authorization import is_authorized
from tests.helpers import ADMIN, cb_update, msg_update


class AuthTests(unittest.TestCase):
    def test_admin_message_and_callback(self):
        self.assertTrue(is_authorized(msg_update(1, text="/start"), ADMIN))
        self.assertTrue(is_authorized(cb_update(2, "swap"), ADMIN))

    def test_other_user_rejected(self):
        self.assertFalse(is_authorized(msg_update(1, user_id=999, text="/start"), ADMIN))
        self.assertFalse(is_authorized(cb_update(2, "cancel", user_id=999), ADMIN))

    def test_group_chat_rejected(self):
        self.assertFalse(is_authorized(msg_update(1, text="/start", chat_type="group"), ADMIN))

    def test_chat_id_mismatch_rejected(self):
        u = msg_update(1, text="/start")
        u["message"]["chat"]["id"] = 77
        self.assertFalse(is_authorized(u, ADMIN))

    def test_bot_sender_and_bool_id_rejected(self):
        u = msg_update(1, text="x")
        u["message"]["from"]["is_bot"] = True
        self.assertFalse(is_authorized(u, ADMIN))
        u = msg_update(1, text="x", user_id=1)
        u["message"]["from"]["id"] = True
        u["message"]["chat"]["id"] = True
        self.assertFalse(is_authorized(u, 1))

    def test_other_update_types_and_garbage_rejected(self):
        self.assertFalse(is_authorized({"update_id": 1, "inline_query": {}}, ADMIN))
        self.assertFalse(is_authorized({"message": "str"}, ADMIN))
        self.assertFalse(is_authorized({"message": {"from": None, "chat": {}}}, ADMIN))
        self.assertFalse(is_authorized({"callback_query": {"from": {"id": ADMIN}}}, ADMIN))

    def test_username_is_not_authentication(self):
        u = msg_update(1, user_id=999, text="x")
        u["message"]["from"]["username"] = "admin"
        self.assertFalse(is_authorized(u, ADMIN))


if __name__ == "__main__":
    unittest.main()
