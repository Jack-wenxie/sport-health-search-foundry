#requires -version 5
# Assertion-based smoke script used by CI.
param()

$ErrorActionPreference = 'Stop'
$fail = 0
function Check($cond,$msg){ if(-not $cond){ Write-Output "FAIL: $msg"; $script:fail++ } else { Write-Output "ok:   $msg" } }
function Resolve-Python {
  if ($env:SHSF_PYTHON -and (Test-Path $env:SHSF_PYTHON)) { return $env:SHSF_PYTHON }
  $py3 = Get-Command python3 -ErrorAction SilentlyContinue
  if ($py3) { return $py3.Source }
  $py = Get-Command python -ErrorAction SilentlyContinue
  if ($py) { return $py.Source }
  throw 'No Python interpreter found. Set SHSF_PYTHON.'
}

$RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
Set-Location $RepoRoot

# 1) Required files exist.
foreach($x in @('README.md','CONTEXT.md','docs/adr/0001-local-bm25-before-opensearch.md','docs/evidence/EVIDENCE_MATRIX.md')){ Check (Test-Path $x) "exists $x" }
# 2) At least one non-README golden fixture exists.
$g = Get-ChildItem 'tests/golden' -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne 'README.md' }
Check (($g | Measure-Object).Count -ge 1) "golden fixture present (non-README)"
# 3) Repository-specific smoke test.
$python = Resolve-Python
Write-Output "python: $python"
& $python scripts/smoke.py
Check ($LASTEXITCODE -eq 0) "repo smoke implemented and passing"

# 4) The smoke test writes the verification record.
Check (Test-Path 'benchmarks/raw/verification-smoke-latest.json') "verification record written"

if($fail -gt 0){ Write-Output "VERIFY FAILED ($fail)"; exit 1 } else { Write-Output "VERIFY OK"; exit 0 }
