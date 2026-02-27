import json
import threading

from journal import CaptionJournal


def test_write_and_read_roundtrip(tmp_journal):
    journal = CaptionJournal(tmp_journal)
    journal.load()
    journal.write("2024/img1.jpg", "a cat on a sofa", "success")
    journal.write("2024/img2.jpg", None, "error_corrupt")
    journal.close()

    # Reload from disk
    journal2 = CaptionJournal(tmp_journal)
    records = journal2.load()
    assert len(records) == 2
    assert records["2024/img1.jpg"] == {"caption": "a cat on a sofa", "status": "success"}
    assert records["2024/img2.jpg"] == {"caption": None, "status": "error_corrupt"}


def test_is_done_after_write(tmp_journal):
    journal = CaptionJournal(tmp_journal)
    journal.load()
    assert not journal.is_done("2024/img1.jpg")
    journal.write("2024/img1.jpg", "caption", "success")
    assert journal.is_done("2024/img1.jpg")
    journal.close()


def test_resume_skips_already_done(tmp_journal):
    # First run: write 2 records
    j1 = CaptionJournal(tmp_journal)
    j1.load()
    j1.write("a.jpg", "cap a", "success")
    j1.write("b.jpg", "cap b", "success")
    j1.close()

    # Second run: load existing, check is_done
    j2 = CaptionJournal(tmp_journal)
    j2.load()
    assert j2.is_done("a.jpg")
    assert j2.is_done("b.jpg")
    assert not j2.is_done("c.jpg")
    j2.close()


def test_malformed_line_tolerance(tmp_journal):
    # Write a valid line, a malformed line, and another valid line
    with open(tmp_journal, "w", encoding="utf-8") as f:
        f.write(json.dumps({"rel_path": "a.jpg", "caption": "ok", "status": "success"}) + "\n")
        f.write("THIS IS NOT JSON\n")
        f.write(json.dumps({"rel_path": "b.jpg", "caption": "ok2", "status": "success"}) + "\n")

    journal = CaptionJournal(tmp_journal)
    records = journal.load()
    assert len(records) == 2
    assert "a.jpg" in records
    assert "b.jpg" in records


def test_last_record_wins(tmp_journal):
    with open(tmp_journal, "w", encoding="utf-8") as f:
        f.write(json.dumps({"rel_path": "a.jpg", "caption": None, "status": "error_api"}) + "\n")
        f.write(json.dumps({"rel_path": "a.jpg", "caption": "fixed", "status": "success"}) + "\n")

    journal = CaptionJournal(tmp_journal)
    records = journal.load()
    assert len(records) == 1
    assert records["a.jpg"] == {"caption": "fixed", "status": "success"}


def test_retry_statuses_exclude_from_done(tmp_journal):
    with open(tmp_journal, "w", encoding="utf-8") as f:
        f.write(json.dumps({"rel_path": "a.jpg", "caption": "ok", "status": "success"}) + "\n")
        f.write(json.dumps({"rel_path": "b.jpg", "caption": None, "status": "error_api"}) + "\n")

    journal = CaptionJournal(tmp_journal)
    journal.load(retry_statuses={"error_api"})
    assert journal.is_done("a.jpg")
    assert not journal.is_done("b.jpg")  # excluded for retry


def test_concurrent_writes_no_interleave(tmp_journal):
    journal = CaptionJournal(tmp_journal)
    journal.load()

    def write_batch(start):
        for i in range(50):
            journal.write(f"img_{start + i}.jpg", f"caption {start + i}", "success")

    t1 = threading.Thread(target=write_batch, args=(0,))
    t2 = threading.Thread(target=write_batch, args=(50,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    journal.close()

    # Verify: 100 valid lines, all parseable
    with open(tmp_journal, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 100
    for line in lines:
        obj = json.loads(line)  # should not raise
        assert "rel_path" in obj


def test_summary(tmp_journal):
    with open(tmp_journal, "w", encoding="utf-8") as f:
        f.write(json.dumps({"rel_path": "a.jpg", "caption": "ok", "status": "success"}) + "\n")
        f.write(json.dumps({"rel_path": "b.jpg", "caption": None, "status": "error_api"}) + "\n")
        f.write(json.dumps({"rel_path": "c.jpg", "caption": "ok2", "status": "success"}) + "\n")

    journal = CaptionJournal(tmp_journal)
    journal.load()
    s = journal.summary()
    assert "3 already processed" in s
    assert "2 success" in s
    assert "1 errors" in s
