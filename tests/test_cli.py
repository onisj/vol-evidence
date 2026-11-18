from volevidence import cli, iv_audit


def test_iv_command_reports_gap(monkeypatch, capsys):
    rows = [{"status": "solved", "gap_vol_pts": g, "kind": "call", "strike": 100.0, "expiry": "2027-01-15",
             "vendor_iv": 0.3, "solved_iv": 0.3 - g / 100} for g in (1.0, 6.0, -2.0)]
    monkeypatch.setattr(iv_audit, "audit_ticker", lambda t, d: rows + [{"status": "below_floor"}])
    assert cli.main(["iv", "SPY"]) == 0
    text = capsys.readouterr().out
    assert "4 two-sided contracts, 3 solvable" in text and "2.00 vol points" in text and "33.3%" in text


def test_iv_command_with_no_quotes(monkeypatch, capsys):
    monkeypatch.setattr(iv_audit, "audit_ticker", lambda t, d: [])
    assert cli.main(["iv", "SPY"]) == 1
