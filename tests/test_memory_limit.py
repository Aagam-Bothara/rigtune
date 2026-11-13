"""Inside a container rigtune budgets host RAM against the cgroup's memory limit, not the host's free RAM."""

from __future__ import annotations

from types import SimpleNamespace

import rigtune.hardware as hwmod

GiB = 1024 ** 3


def test_cgroup_v2_limit_less_working_set(tmp_path):
    (tmp_path / "memory.max").write_text(f"{60 * GiB}\n")
    (tmp_path / "memory.current").write_text(f"{20 * GiB}\n")
    # File cache, active and inactive, is reclaimed before the limit is enforced: it does not count as used.
    (tmp_path / "memory.stat").write_text(f"anon {12 * GiB}\nfile {8 * GiB}\nactive_file {3 * GiB}\n"
                                          f"inactive_file {5 * GiB}\n")
    assert hwmod._memory_limit(str(tmp_path)) == 48 * GiB
    (tmp_path / "memory.max").write_text("max\n")
    assert hwmod._memory_limit(str(tmp_path)) is None


# A cgroup v2 memory.stat in the kernel's format, as a 60 GiB pod might show it just after downloading 16 GiB
# of weights: most of its usage is page cache, the active part as large as the inactive.
REALISTIC_STAT = """anon 5583457280
file 17716740096
kernel 268435456
kernel_stack 9437184
pagetables 41943040
sec_pagetables 0
percpu 1048576
sock 4096
vmalloc 65536
shmem 2097152
file_mapped 1073741824
file_dirty 8388608
file_writeback 0
swapcached 0
anon_thp 2097152000
inactive_anon 5368709120
active_anon 216748032
inactive_file 7945689088
active_file 9771051008
unevictable 0
slab_reclaimable 157286400
slab_unreclaimable 52428800
slab 209715200
workingset_refault_anon 0
pgfault 41256731
"""


def test_a_fresh_weight_download_does_not_fill_the_container(tmp_path):
    (tmp_path / "memory.max").write_text(f"{60 * GiB}\n")
    (tmp_path / "memory.current").write_text(f"{23568842752}\n")  # about anon + file + kernel
    (tmp_path / "memory.stat").write_text(REALISTIC_STAT)
    room = hwmod._memory_limit(str(tmp_path))
    assert room == 60 * GiB - (23568842752 - 7945689088 - 9771051008)  # 54.5 GiB
    # Leaving the active file cache in, as the kubelet's working set does, would report 9.1 GiB less.
    assert room - (60 * GiB - (23568842752 - 7945689088)) == 9771051008


def test_cgroup_v2_without_usage_files_counts_the_whole_limit(tmp_path):
    (tmp_path / "memory.max").write_text(f"{8 * GiB}\n")
    assert hwmod._memory_limit(str(tmp_path)) == 8 * GiB


def test_cgroup_v2_over_the_limit_leaves_nothing(tmp_path):
    (tmp_path / "memory.max").write_text(f"{8 * GiB}\n")
    (tmp_path / "memory.current").write_text(f"{9 * GiB}\n")
    assert hwmod._memory_limit(str(tmp_path)) == 0


def test_cgroup_v1_limit(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.limit_in_bytes").write_text(f"{32 * GiB}\n")
    (mem / "memory.usage_in_bytes").write_text(f"{10 * GiB}\n")
    # v1 lists this cgroup's own counts and the hierarchy's (total_*); the hierarchy's are what count.
    (mem / "memory.stat").write_text(f"cache {4 * GiB}\nactive_file {7}\ninactive_file {9}\n"
                                     f"total_active_file {1 * GiB}\ntotal_inactive_file {2 * GiB}\n")
    assert hwmod._memory_limit(str(tmp_path)) == 25 * GiB
    # cgroup v1 spells "no limit" as a page-rounded 2**63 - 1.
    (mem / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert hwmod._memory_limit(str(tmp_path)) is None


def test_no_cgroup_or_garbage_is_unknown(tmp_path):
    assert hwmod._memory_limit(str(tmp_path / "missing")) is None
    (tmp_path / "memory.max").write_text("lots\n")
    assert hwmod._memory_limit(str(tmp_path)) is None


def _vm(total: int, available: int):
    return SimpleNamespace(total=total, available=available)


def test_probe_caps_free_ram_to_the_container_limit(monkeypatch):
    monkeypatch.setattr(hwmod.psutil, "virtual_memory", lambda: _vm(1024 * GiB, 900 * GiB))
    monkeypatch.setattr(hwmod, "_usable_cpus", lambda: None)
    monkeypatch.setattr(hwmod, "_memory_limit", lambda: 45 * GiB)
    c = hwmod.probe_cpu()
    assert c.ram_free_bytes == 45 * GiB
    assert c.ram_total_bytes == 1024 * GiB  # the hardware hash keeps the host's total
    monkeypatch.setattr(hwmod, "_memory_limit", lambda: None)  # no container limit: unchanged
    assert hwmod.probe_cpu().ram_free_bytes == 900 * GiB
    monkeypatch.setattr(hwmod, "_memory_limit", lambda: 2000 * GiB)  # a limit above free RAM changes nothing
    assert hwmod.probe_cpu().ram_free_bytes == 900 * GiB
