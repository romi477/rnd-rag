from rag.traceback import looks_like_traceback, parse_traceback, semantic_query, version_note

MODULES = {"demo_core", "demo_shop"}

TB = '''Traceback (most recent call last):
  File "/opt/external-addons/queue_lib/queue_job/controllers/main.py", line 113, in runjob
    self._try_perform_job(env, job)
  File "/opt/project/demo_core/models/order.py", line 19, in action_sync
    return self.client.get_orders(since)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/project/demo_shop/api/client.py", line 7, in get_orders
    raise ShopApiError(response.text, response.status_code)
odoo.addons.demo_shop.client.exceptions.ShopApiError: {"code":"bad","message":"Invalid parameter(s): type"}
'''

USER_ERROR_TB = '''Traceback (most recent call last):
  File "/opt/project/demo_core/models/pipeline.py", line 382, in _run_and_call_next
    raise es.UserError(message)
odoo.exceptions.UserError: Sales Order (id=5) [#100]: [WH/OUT/00001]: Transfer trouble alert!
Set some quantities and let's get moving!

'''


def test_frames_have_path_line_function_and_code_without_caret_lines():
    tb = parse_traceback(TB, MODULES)
    assert [(f.line, f.func) for f in tb.frames] == [(113, "runjob"), (19, "action_sync"), (7, "get_orders")]
    assert tb.frames[1].code == "return self.client.get_orders(since)"
    assert tb.frames[2].code == "raise ShopApiError(response.text, response.status_code)"


def test_frames_resolve_to_modules_by_directory_name():
    tb = parse_traceback(TB, MODULES)
    assert [(f.module, f.rel) for f in tb.frames] == [
        ("", ""), ("demo_core", "models/order.py"), ("demo_shop", "api/client.py"),
    ]


def test_module_is_found_whatever_the_server_prefix_and_leftmost_wins():
    text = 'File "/home/odoo/src/custom/demo_core/demo_core/a.py", line 3, in f\n    x = 1\nValueError: boom\n'
    (frame,) = parse_traceback(text, MODULES).frames
    assert (frame.module, frame.rel) == ("demo_core", "demo_core/a.py")


def test_exception_name_and_message():
    exc = parse_traceback(TB, MODULES).exception
    assert exc.name == "odoo.addons.demo_shop.client.exceptions.ShopApiError"
    assert exc.short_name == "ShopApiError"
    assert exc.message == '{"code":"bad","message":"Invalid parameter(s): type"}'


def test_multiline_message_is_kept_whole_without_trailing_blank_lines():
    exc = parse_traceback(USER_ERROR_TB, MODULES).exception
    assert exc.name == "odoo.exceptions.UserError"
    assert exc.message == "Sales Order (id=5) [#100]: [WH/OUT/00001]: Transfer trouble alert!\nSet some quantities and let's get moving!"


def test_own_frames_are_innermost_first_and_skip_foreign_ones():
    assert [f.func for f in parse_traceback(TB, MODULES).own_frames] == ["get_orders", "action_sync"]


def test_truncated_traceback_without_header_or_exception_line():
    text = '  File "/opt/project/demo_core/models/order.py", line 19, in action_sync\n    return 1\n'
    tb = parse_traceback(text, MODULES)
    assert [f.func for f in tb.frames] == ["action_sync"] and tb.exception is None


def test_line_wrapped_frame_header_and_path():
    text = 'File "/opt/project/demo_core/mod\nels/order.py",\n line 19, in action_sync\n    return 1\nKeyError: 1\n'
    (frame,) = parse_traceback(text, MODULES).frames
    assert (frame.rel, frame.line, frame.func) == ("models/order.py", 19, "action_sync")


def test_crlf_line_endings():
    tb = parse_traceback(TB.replace("\n", "\r\n"), MODULES)
    assert len(tb.frames) == 3 and tb.exception.short_name == "ShopApiError"


def test_chained_exceptions_main_is_the_last_one():
    text = (
        'Traceback (most recent call last):\n  File "/opt/project/demo_core/a.py", line 2, in f\n    d[1]\nKeyError: 1\n\n'
        "During handling of the above exception, another exception occurred:\n\n" + TB
    )
    tb = parse_traceback(text, MODULES)
    assert tb.exception.short_name == "ShopApiError"
    assert [f.func for f in tb.frames] == ["runjob", "action_sync", "get_orders"]
    assert [(c.exception.name, [f.func for f in c.frames]) for c in tb.chained] == [("KeyError", ["f"])]


def test_two_tracebacks_pasted_one_after_another():
    tb = parse_traceback(USER_ERROR_TB + TB, MODULES)
    assert tb.exception.short_name == "ShopApiError" and len(tb.chained) == 1


def test_error_codes_come_from_message_and_raise_site_not_from_longer_numbers():
    text = (
        'File "/opt/project/demo_core/a.py", line 2, in f\n    ErrorStore.raise_error(err_code=\'E514\')\n'
        "odoo.exceptions.UserError: [E105] adapter missing; id E1234 and ABE123x are not codes\n"
    )
    assert parse_traceback(text, MODULES).error_codes == ["E105", "E514"]


def test_bare_exception_line_without_frames():
    tb = parse_traceback("odoo.exceptions.UserError: Something is off", MODULES)
    assert tb.frames == [] and tb.exception.message == "Something is off"


def test_plain_text_is_not_a_traceback():
    tb = parse_traceback("how do I configure the order import?", MODULES)
    assert tb.frames == [] and tb.exception is None
    assert not looks_like_traceback("how do I configure the order import?")


def test_looks_like_traceback():
    assert looks_like_traceback(TB)
    assert looks_like_traceback('  File "/x/y.py", line 1, in f\n    pass\n')
    assert looks_like_traceback("odoo.exceptions.UserError: boom")


def test_semantic_query_has_message_exception_and_own_function_names():
    query = semantic_query(parse_traceback(TB, MODULES))
    assert "Invalid parameter(s): type" in query
    assert "ShopApiError" in query and "get_orders" in query and "action_sync" in query
    assert "runjob" not in query
    assert len(semantic_query(parse_traceback(TB, MODULES), max_chars=40)) <= 40


def test_version_note_only_when_versions_differ():
    assert version_note(None, "19.0.2.3.0") is None
    assert version_note("19.0.2.3.0", "19.0.2.3.0") is None
    note = version_note("19.0.2.2.0", "19.0.2.3.0")
    assert "19.0.2.2.0" in note and "19.0.2.3.0" in note and "line numbers" in note


def test_json_message_is_reduced_to_its_readable_part():
    text = (
        'File "/opt/project/demo_shop/a.py", line 2, in f\n    x\n'
        'ShopApiError: {"code":"rest_invalid_param","message":"Invalid parameter(s): type","data":{"status":400,"params":{"type":"long"}}}\n'
    )
    query = semantic_query(parse_traceback(text, MODULES))
    assert query.startswith("rest_invalid_param Invalid parameter(s): type")
    assert "status" not in query and "params" not in query
