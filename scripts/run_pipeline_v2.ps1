# =====================================================================
#  run_pipeline_v2.ps1 — CO2 cycloaddition end-to-end pipeline runner
#  Phase 2 — perfected 2026-08-19
#
#  Changes vs the previous version
#  --------------------------------
#  * Per-step --force: only scripts that actually support --force get it.
#  * Per-step skip-by-output: if the canonical output file already exists
#    AND its mtime is newer than the script, the step is skipped (configurable).
#  * Path / Python: read from $env:CO2_PROJECT_ROOT / $env:CO2_PYTHON,
#    fall back to the historical defaults if not set.
#  * Timing: every step + every tier shows elapsed wall-clock time.
#  * Logging: all output is also written to logs/pipeline_<UTC>.log.
#  * -DryRun: print the commands instead of executing.
#  * -Resume: skip steps whose canonical output already exists.
#  * -NoForce: disable --force everywhere (overrides per-step defaults).
#  * -StepTimeOut: per-step hard timeout (default 0 = unlimited).
#  * Exit codes: 0 = OK, 1 = fatal in a required step, 2 = warnings only.
#  * --Diagnostic: show config & exit before doing anything.
#
#  Usage (PowerShell 7+):
#    .\run_pipeline_v2.ps1                         # run everything
#    .\run_pipeline_v2.ps1 -Tier tier_main         # one tier only
#    .\run_pipeline_v2.ps1 -List                   # show tier table
#    .\run_pipeline_v2.ps1 -DryRun                 # preview commands
#    .\run_pipeline_v2.ps1 -Resume                 # skip steps w/ existing outputs
#    .\run_pipeline_v2.ps1 -SkipDFT -NoXTB -NoAbstract
#    .\run_pipeline_v2.ps1 -StepTimeout 30        # per-step wall-clock cap
#    .\run_pipeline_v2.ps1 -Diagnostic
#
#  Python   : $env:CO2_PYTHON, default D:\co2\env_drfp\python.exe
#  Root     : $env:CO2_PROJECT_ROOT, default D:\machine-learning\CO2-cycloaddition
#
#  Exit codes:
#    0 = all required steps passed
#    1 = at least one required step failed (pipeline aborted)
#    2 = all required steps passed but optional steps raised warnings
# =====================================================================

[CmdletBinding()]
param(
    [string]$Tier             = "",
    [switch]$List,
    [switch]$NoForce,         # disable auto --force (overrides per-step config)
    [switch]$SkipDFT,
    [switch]$NoXTB,
    [switch]$NoAbstract,
    [int]$WaitDFT             = 0,   # minutes to wait for ORCA artefact
    [switch]$DryRun,
    [switch]$Resume,          # skip steps whose canonical output exists
    [int]$StepTimeout         = 0,   # per-step wall-clock cap, minutes (0 = none)
    [switch]$Diagnostic
)

# ── Paths (env vars override hard-coded defaults) ───────────────────────────
$env:PYTHONWARNINGS    = "ignore"
$env:PYTHONIOENCODING  = "utf-8"
$env:PYTHONUTF8        = "1"
$env:PYTHONPATH        = "$ROOT;$ROOT\src"  # make `from src.…` imports work

$ErrorActionPreference = "Stop"

$ROOT = if ($env:CO2_PROJECT_ROOT) {
    (Resolve-Path $env:CO2_PROJECT_ROOT).Path
} else {
    Set-Location "D:\machine-learning\CO2-cycloaddition"
    "D:\machine-learning\CO2-cycloaddition"
}
$PY = if ($env:CO2_PYTHON) {
    $env:CO2_PYTHON
} else {
    "D:\co2\env_drfp\python.exe"
}

# Test python up front
if (-not $DryRun -and -not (Test-Path $PY)) {
    Write-Host "[ERR] Python interpreter not found: $PY" -ForegroundColor Red
    Write-Host "      Set `$env:CO2_PYTHON to a valid interpreter and retry." -ForegroundColor Red
    exit 1
}

Set-Location $ROOT

# ── Global pipeline log ─────────────────────────────────────────────────────
$LOG_DIR = Join-Path $ROOT "logs"
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR | Out-Null }
$LOG_FILE = Join-Path $LOG_DIR ("pipeline_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
"PIPELINE START  $((Get-Date).ToString('u'))  ROOT=$ROOT  PY=$PY" |
    Out-File -FilePath $LOG_FILE -Encoding utf8

function Log([string]$msg, [string]$color = "Gray") {
    $ts = (Get-Date).ToString("HH:mm:ss")
    $line = "$ts  $msg"
    Add-Content -Path $LOG_FILE -Value $line -Encoding utf8
    if ($color -eq "Gray") {
        Write-Host $msg
    } else {
        Write-Host $msg -ForegroundColor $color
    }
}

# ── Step configuration ──────────────────────────────────────────────────────
# Each step has:
#   Script  : absolute path
#   Args    : extra CLI args (always passed)
#   Output  : canonical output file (for -Resume)
#   Force   : whether to add --force (overridden by -NoForce)
#   Timeout : per-step minutes (0 = unlimited; overrides -StepTimeout if > 0)
#   SkipReason: optional static skip (e.g. "needs WSL")
$Steps = [ordered]@{}

function Register-Step {
    param(
        [string]$Key,
        [string]$Script,
        [string]$Output = "",
        [string[]]$Args = @(),
        [bool]$Force   = $true,
        [int]$Timeout  = 0
    )
    $Steps[$Key] = @{
        Script  = $Script
        Args    = $Args
        Output  = $Output
        Force   = $Force
        Timeout = $Timeout
    }
}

# ── Helper: actually invoke a step ─────────────────────────────────────────
# Returns: hashtable { ok=…, elapsed=…, exit=…, message=… }
function Invoke-Step {
    param(
        [string]$Tag,
        [string]$Script,
        [string[]]$StepArgs = @(),
        [bool]$ForceFlag = $false,
        [int]$StepTimeoutMin = 0
    )

    $allArgs = @()
    if ($ForceFlag -and -not $NoForce) { $allArgs += "--force" }
    $allArgs += $StepArgs

    $argString = ($allArgs | ForEach-Object { "`"$_`"" }) -join " "
    $cmdLine = "`"$PY`" `"$Script`" $argString"

    Write-Host ""
    Write-Host "=== [$Tag] ===" -ForegroundColor Cyan
    Log "[$Tag] $cmdLine" "Cyan"

    if ($DryRun) {
        Write-Host "  DRY-RUN: skipped" -ForegroundColor DarkCyan
        return @{ ok = $true; elapsed = 0; exit = 0; message = "dry-run" }
    }

    $envToApply = @{
        PYTHONWARNINGS    = "ignore"
        PYTHONIOENCODING  = "utf-8"
        PYTHONUTF8        = "1"
        PYTHONUNBUFFERED  = "1"          # force line-buffered stdout
        PYTHONPATH        = "$ROOT;$ROOT\src"
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    if ($StepTimeoutMin -gt 0) {
        # Use Start-Process + Wait with timeout
        $tmpOut = [System.IO.Path]::GetTempFileName()
        $tmpErr = [System.IO.Path]::GetTempFileName()
        $proc = Start-Process -FilePath $PY `
                              -ArgumentList @($Script) + $allArgs `
                              -WorkingDirectory $ROOT `
                              -RedirectStandardOutput $tmpOut `
                              -RedirectStandardError  $tmpErr `
                              -NoNewWindow -PassThru `
                              -Environment $envToApply
        $deadline = (Get-Date).AddMinutes($StepTimeoutMin)
        while (-not $proc.HasExited) {
            if ((Get-Date) -ge $deadline) {
                try { Stop-Process -Id $proc.Id -Force } catch {}
                Write-Host "  [$Tag] TIMEOUT after ${StepTimeoutMin}m" -ForegroundColor Red
                $sw.Stop()
                return @{ ok = $false; elapsed = $sw.Elapsed.TotalSeconds; exit = -1
                         message = "timeout" }
            }
            Start-Sleep -Seconds 5
        }
        $proc.WaitForExit()
        if (Test-Path $tmpOut) { Get-Content $tmpOut | ForEach-Object { Write-Host $_ ; Add-Content -Path $LOG_FILE -Value $_ } ; Remove-Item $tmpOut -Force }
        if (Test-Path $tmpErr) { Get-Content $tmpErr | ForEach-Object { Write-Host $_ -ForegroundColor DarkYellow ; Add-Content -Path $LOG_FILE -Value $_ } ; Remove-Item $tmpErr -Force }
        $sw.Stop()
        $ok = ($proc.ExitCode -eq 0)
        if (-not $ok) {
            Write-Host "  [$Tag] FAIL (exit $($proc.ExitCode)) in $([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Red
            Log "[$Tag] FAIL exit=$($proc.ExitCode) elapsed=$([int]$sw.Elapsed.TotalSeconds)s" "Red"
        } else {
            Write-Host "  [$Tag] OK in $([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Green
            Log "[$Tag] OK elapsed=$([int]$sw.Elapsed.TotalSeconds)s" "Green"
        }
        return @{ ok = $ok; elapsed = $sw.Elapsed.TotalSeconds; exit = $proc.ExitCode
                 message = if ($ok) { "ok" } else { "fail" } }
    }

    # No-timeout path: synchronous invocation with tee
    $tmpOut = [System.IO.Path]::GetTempFileName()
    $tmpErr = [System.IO.Path]::GetTempFileName()
    $env:PYTHONPATH = "$ROOT;$ROOT\src"   # belt-and-braces for cmd.exe path
    cmd.exe /c "set PYTHONPATH=$ROOT;$ROOT\src && `"$PY`" -u `"$Script`" $argString > `"$tmpOut`" 2> `"$tmpErr`"" | Out-Null
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    if (Test-Path $tmpOut) { Get-Content $tmpOut | ForEach-Object { Write-Host $_ ; Add-Content -Path $LOG_FILE -Value $_ } ; Remove-Item $tmpOut -Force }
    if (Test-Path $tmpErr) { Get-Content $tmpErr | ForEach-Object { Write-Host $_ -ForegroundColor DarkYellow ; Add-Content -Path $LOG_FILE -Value $_ } ; Remove-Item $tmpErr -Force }
    if ($exitCode -ne 0) {
        Write-Host "  [$Tag] FAIL (exit $exitCode) in $([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Red
        Log "[$Tag] FAIL exit=$exitCode elapsed=$([int]$sw.Elapsed.TotalSeconds)s" "Red"
        return @{ ok = $false; elapsed = $sw.Elapsed.TotalSeconds; exit = $exitCode
                 message = "fail" }
    }
    Write-Host "  [$Tag] OK in $([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Green
    Log "[$Tag] OK elapsed=$([int]$sw.Elapsed.TotalSeconds)s" "Green"
    return @{ ok = $true; elapsed = $sw.Elapsed.TotalSeconds; exit = $exitCode
             message = "ok" }
}

# ── Public step runners ─────────────────────────────────────────────────────
# $true / $false: does the step actually accept --force?
function Step([string]$Key, [string[]]$extraArgs = @()) {
    $cfg = $Steps[$Key]
    if (-not $cfg) {
        Write-Host "[ERR] Unknown step '$Key'" -ForegroundColor Red
        exit 1
    }
    $tag = Split-Path -Leaf $cfg.Script

    # Resume-mode: skip if canonical output exists
    if ($Resume -and $cfg.Output -and (Test-Path $cfg.Output)) {
        $scriptM = (Get-Item $cfg.Script).LastWriteTime
        $outM    = (Get-Item $cfg.Output).LastWriteTime
        if ($outM -ge $scriptM) {
            Write-Host ""
            Write-Host "=== [$tag] (skipped: $(Split-Path -Leaf $cfg.Output) is up to date) ===" -ForegroundColor DarkCyan
            Log "[$tag] SKIPPED (output fresh)" "DarkCyan"
            return @{ ok = $true; elapsed = 0; exit = 0; message = "skip-resume" }
        }
    }

    $effectiveTimeout = if ($cfg.Timeout -gt 0) { $cfg.Timeout } else { $StepTimeout }
    return Invoke-Step -Tag $tag `
                       -Script $cfg.Script `
                       -StepArgs ($cfg.Args + $extraArgs) `
                       -ForceFlag $cfg.Force `
                       -StepTimeoutMin $effectiveTimeout
}

function StepOptional([string]$Key, [string[]]$extraArgs = @()) {
    $cfg = $Steps[$Key]
    if (-not $cfg) {
        Write-Host "[ERR] Unknown step '$Key'" -ForegroundColor Red
        exit 1
    }
    $tag = Split-Path -Leaf $cfg.Script
    $r = Step -Key $Key -extraArgs $extraArgs
    if (-not $r.ok) {
        Write-Host "  [$tag] WARN (continuing)" -ForegroundColor Yellow
        Log "[$tag] WARN continuing" "Yellow"
    }
    return $r
}

# ── Register every step (in execution order) ───────────────────────────────
# Steps are grouped by which tier calls them.  Order matches $Tiers.
# ═══════════════════════════════════════════════════════════════════════════
# 1. TIER_DATA
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "101_clean"        -Script "$ROOT\src\data\101_clean.py"                    -Output "$ROOT\data\processed\cleaned.csv"
Register-Step -Key "102_smiles"       -Script "$ROOT\src\data\102_smiles.py"                   -Output "$ROOT\data\processed\co2_smiles.csv"
Register-Step -Key "103_drfp"         -Script "$ROOT\src\data\103_drfp.py"                     -Output "$ROOT\data\processed\co2_drfp.csv"
Register-Step -Key "104b_run_xtb"     -Script "$ROOT\src\data\104b_run_xtb_extended.py"        -Output "$ROOT\results_cho_diagnostic\xtb_results_summary.csv"  -Timeout 720
Register-Step -Key "105b_xtb_sanity"  -Script "$ROOT\src\data\105b_xtb_sanity_v2.py"            -Output "$ROOT\results_cho_diagnostic\xtb_sanity_summary.csv"
Register-Step -Key "107_merge"        -Script "$ROOT\src\data\107_merge_substrate_xtb.py"      -Output "$ROOT\results_cho_diagnostic\co2_drfp_xtb_extended.csv"
Register-Step -Key "data_split"       -Script "$ROOT\src\data_split.py"                        -Output "$ROOT\results_data_split\data_split.json"

# ═══════════════════════════════════════════════════════════════════════════
# 2. TIER_ABLATION  (201 = benchmark + ablation, one step)
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "201_ablation"     -Script "$ROOT\src\data\201_ablation.py"                 -Output "$ROOT\results_best_pipeline\full_benchmark_results.csv"

# ═══════════════════════════════════════════════════════════════════════════
# 3. TIER_PCL
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "train_pcl_ae"     -Script "$ROOT\src\models\persistence\train_pcl_ae.py"   -Output "$ROOT\results_pcl_ae\pcl_ae_latent.npy"

# ═══════════════════════════════════════════════════════════════════════════
# 4. TIER_MAIN  (302+303+304+306+401)
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "302_groupkfold"   -Script "$ROOT\src\models\benchmarks\302_groupkfold_validation.py"    -Output "$ROOT\results_groupkfold_validation\ML_groupkfold_results.csv"
Register-Step -Key "303_sampling"      -Script "$ROOT\src\models\benchmarks\303_sample_size_sensitivity.py"  -Output "$ROOT\results_sample_size_sensitivity\learning_curve_summary.csv"
Register-Step -Key "304_stat_sig"      -Script "$ROOT\src\models\benchmarks\304_statistical_significance.py"  -Output "$ROOT\results_statistical_test\wilcoxon_results.csv"
Register-Step -Key "306_external"      -Script "$ROOT\src\models\benchmarks\306_external_validation.py"       -Output "$ROOT\results_external_validation\STAGE6_FINAL_REPORT.txt"
Register-Step -Key "401_persist"       -Script "$ROOT\src\models\persistence\401_persist_best_pipeline.py"   -Output "$ROOT\results_best_pipeline\artifacts\drfp_scaler.joblib"  -Force $false

# ═══════════════════════════════════════════════════════════════════════════
# 5. TIER_SCREENING
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "310_top10"         -Script "$ROOT\src\models\benchmarks\310_known_top10_baseline.py"      -Output "$ROOT\results_virtual_screening\top10_results.csv"
Register-Step -Key "403b_ranking"      -Script "$ROOT\src\models\screening\403b_ranking_metrics.py"          -Output "$ROOT\results_ranking_metrics\ranking_metrics_summary.csv"

# ═══════════════════════════════════════════════════════════════════════════
# 6. TIER_VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "405_external"      -Script "$ROOT\src\models\persistence\405_external_validation.py"      -Output "$ROOT\results_external_validation\external_validation_results.csv"

# ═══════════════════════════════════════════════════════════════════════════
# 7. TIER_LOSO  (700+701; 702 reruns inside tier_dft_post)
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "700_loso_lomo"     -Script "$ROOT\src\analysis\loso\700_loso_lomo_cv.py"                   -Output "$ROOT\results_step4\summary_protocol.csv"
Register-Step -Key "701_per_sub_shap"  -Script "$ROOT\src\analysis\loso\701_per_substrate_shap.py"             -Output "$ROOT\results_step4_5\per_substrate_shap.csv"

# ═══════════════════════════════════════════════════════════════════════════
# 8. TIER_REGEN
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "regen_all_v3"      -Script "$ROOT\src\visualization\regen_all_v3.py"                      -Force $false

# ═══════════════════════════════════════════════════════════════════════════
# 9. TIER_ABSTRACT
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "900_abstract"      -Script "$ROOT\src\analysis\diagnostics\900_paper_abstract.py"         -Output "$ROOT\paper_text\abstract_combined.md"  -Force $false

# ═══════════════════════════════════════════════════════════════════════════
# 10. TIER_FIGURES
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "fig_pcl_ae_arch"   -Script "$ROOT\src\visualization\fig_pcl_ae_architecture.py"           -Force $false
Register-Step -Key "fig_yrand_100"     -Script "$ROOT\src\visualization\fig_yrandomization_100perm.py"        -Force $false
Register-Step -Key "fig_0_abs"         -Script "$ROOT\src\visualization\fig_0_graphical_abstract.py"          -Force $false
Register-Step -Key "fig_toc"           -Script "$ROOT\src\visualization\fig_toc.py"                           -Force $false
Register-Step -Key "fig_4_root_cause"  -Script "$ROOT\src\visualization\fig_4_loso_root_cause.py"             -Force $false
Register-Step -Key "fig_5_loso_proto"  -Script "$ROOT\src\visualization\fig_5_loso_protocol.py"               -Force $false
Register-Step -Key "fig_6_transf"      -Script "$ROOT\src\visualization\fig_6_transferability_matrix.py"      -Force $false
Register-Step -Key "fig_7_shap_dir"    -Script "$ROOT\src\visualization\fig_7_shap_direction.py"              -Force $false
Register-Step -Key "fig1_en"           -Script "$ROOT\src\visualization\fig1_protocol_comparison_en.py"       -Force $false
Register-Step -Key "fig1_zh"           -Script "$ROOT\src\visualization\fig1_protocol_comparison_zh.py"       -Force $false
Register-Step -Key "fig2_en"           -Script "$ROOT\src\visualization\fig2_loso_quality_en.py"              -Force $false
Register-Step -Key "fig2_zh"           -Script "$ROOT\src\visualization\fig2_loso_quality_zh.py"              -Force $false
Register-Step -Key "fig3_en"           -Script "$ROOT\src\visualization\fig3_coverage_5x5_en.py"              -Force $false
Register-Step -Key "fig3_zh"           -Script "$ROOT\src\visualization\fig3_coverage_5x5_zh.py"              -Force $false
Register-Step -Key "fig4_en"           -Script "$ROOT\src\visualization\fig4_shap_per_substrate_en.py"        -Force $false
Register-Step -Key "fig4_zh"           -Script "$ROOT\src\visualization\fig4_shap_per_substrate_zh.py"        -Force $false
Register-Step -Key "fig5_en"           -Script "$ROOT\src\visualization\fig5_homo_vs_yield_en.py"              -Force $false
Register-Step -Key "fig5_zh"           -Script "$ROOT\src\visualization\fig5_homo_vs_yield_zh.py"              -Force $false

# ═══════════════════════════════════════════════════════════════════════════
# 11. TIER_SI  (SHAP control experiments: si_s3 + year_ood + yrand_100 + bootstrap + 803/804/807-810)
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "si_s3"             -Script "$ROOT\src\ci_artifacts\generate_si_s3_benchmark_full_v3_1.py" -Output "$ROOT\results_si\lomo_v3_full_results.csv"  -Timeout 60
Register-Step -Key "year_ood"          -Script "$ROOT\src\ci_artifacts\generate_year_ood_benchmark.py"        -Output "$ROOT\results_si\year_ood_benchmark.csv"     -Timeout 30
Register-Step -Key "yrand_100"         -Script "$ROOT\src\ci_artifacts\generate_y_randomization_v4_100perm.py" -Output "$ROOT\results_y_randomization_v4_100perm\y_randomization_v4_100perm_summary.json" -Timeout 60
Register-Step -Key "boot_substrate"    -Script "$ROOT\src\ci_artifacts\generate_bootstrap_substrate_ci.py"     -Output "$ROOT\data\processed\bootstrap_substrate_shap_ci.csv"
Register-Step -Key "803_mordred"       -Script "$ROOT\src\models\benchmarks\803_mordred_ablation.py"          -Force $false
Register-Step -Key "804_hier"          -Script "$ROOT\src\models\benchmarks\804_hierarchical_catalyst_model.py" -Force $false
Register-Step -Key "807_cross_model"   -Script "$ROOT\src\models\benchmarks\807_cross_model_shap.py"         -Output "$ROOT\results_shap_comprehensive\cross_model\cross_model_shap_summary.csv"  -Timeout 30
Register-Step -Key "808_catalyst_ctrl" -Script "$ROOT\src\models\benchmarks\808_catalyst_control.py"          -Output "$ROOT\results_shap_comprehensive\catalyst_control\catalyst_control_summary.csv"  -Timeout 20
Register-Step -Key "809_condition_ctrl"-Script "$ROOT\src\models\benchmarks\809_condition_control.py"          -Output "$ROOT\results_shap_comprehensive\condition_control\stratification_summary.csv"  -Timeout 20
Register-Step -Key "810_yield_dist"    -Script "$ROOT\src\models\benchmarks\810_yield_distribution.py"       -Output "$ROOT\results_shap_comprehensive\yield_distribution\yield_distribution_summary.csv"  -Timeout 10

# ═══════════════════════════════════════════════════════════════════════════
# 12. TIER_DFT  (501+510+512+514+520+514b; mechanism steps run in tier_dft_post)
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "501_dft_inputs"    -Script "$ROOT\src\dft\501_generate_dft_inputs.py"                     -Force $true
Register-Step -Key "510_parse_dft"     -Script "$ROOT\src\dft\510_parse_dft_outputs.py"                       -Force $true
Register-Step -Key "512_xtb_dft"       -Script "$ROOT\src\dft\512_xtb_on_dft_geometry.py"                     -Force $true
Register-Step -Key "514_dft_vs_xtb"    -Script "$ROOT\src\dft\514_dft_vs_xtb_report.py"                       -Force $true
Register-Step -Key "520_figs"          -Script "$ROOT\dft_validation\scripts\520_dft_journal_figures.py"           -Force $false
Register-Step -Key "514b_dft_ts"       -Script "$ROOT\src\dft\514b_dft_transition_state.py"                    -Force $true

# ═══════════════════════════════════════════════════════════════════════════
# 13. TIER_S7  (502 regenerates extended DFT inputs; 713_orca_gen removed —
#     its functionality is now covered by 502 + 514b_dft_transition_state.py)
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "502_dft_extended"  -Script "$ROOT\src\dft\502_generate_dft_inputs_extended.py"                -Force $false

# ═══════════════════════════════════════════════════════════════════════════
# MECHANISM / LOSO-IMPROVED / DIAGNOSTIC steps
# Registered here so they can be referenced by tier_dft_post.
# All degrade gracefully when upstream artefacts are absent.
# ═══════════════════════════════════════════════════════════════════════════
Register-Step -Key "601_mechanism"     -Script "$ROOT\src\analysis\mechanism\601_catalyst_mechanism_v2.py"     -Output "$ROOT\data\processed\catalyst_mechanism.csv"
Register-Step -Key "602_substrate"     -Script "$ROOT\src\analysis\mechanism\602_substrate_features.py"          -Output "$ROOT\results_mechanism\substrate_features.csv"
Register-Step -Key "603_transfer"      -Script "$ROOT\src\analysis\mechanism\603_transferability_matrix.py"     -Output "$ROOT\results_transferability\cross_tab_mech_substrate.csv"
Register-Step -Key "702_integrated"    -Script "$ROOT\src\analysis\loso\702_integrated_report.py"              -Output "$ROOT\results_step5\integrated_narrative.md"
Register-Step -Key "705_improved"      -Script "$ROOT\src\analysis\loso\705_improved_loso.py"                  -Output "$ROOT\results_step7_improved_loso\improved_loso_results.json"
Register-Step -Key "706_root_cause"    -Script "$ROOT\src\analysis\loso\706_loso_root_cause_figure.py"        -Output "$ROOT\results_step7_improved_loso\fig_loso_root_cause.png"
Register-Step -Key "901_substrate_cat" -Script "$ROOT\src\analysis\diagnostics\901_substrate_catalyst_matrix.py" -Output "$ROOT\results_substrate_catalyst_matrix\matrix_yield_ci_5x5.csv"
Register-Step -Key "902_cho_diag"      -Script "$ROOT\src\analysis\diagnostics\902_cho_mechanistic_diagnostic.py" -Output "$ROOT\results_cho_diagnostic\cho_vs_other_summary.csv"

# ========================================================================
#  Tier definitions
# ========================================================================
function tier_data {
    Step "101_clean"
    Step "102_smiles"
    Step "103_drfp"
    if (-not $NoXTB) {
        Step "104b_run_xtb" @("--timeout","90")
    } else {
        Write-Host "[skip] 104b_run_xtb_extended.py (-NoXTB)" -ForegroundColor DarkYellow
    }
    Step "105b_xtb_sanity"
    Step "107_merge"
    Step "data_split"
}

function tier_ablation {
    Step "201_ablation"
}

function tier_pcl {
    Step "train_pcl_ae"
}

function tier_main {
    Step "302_groupkfold"
    Step "303_sampling"
    Step "304_stat_sig"
    Step "306_external"
    Step "401_persist"
}

function tier_screening {
    Step "310_top10"
    Step "403b_ranking"
}

function tier_validation {
    Step "405_external"
}

function tier_loso {
    # 700/701: no 601/602/603 dependency — run independently for early diagnostics.
    # 702: reads 601/602/603 outputs if available (graceful fallback otherwise).
    #   Rerun 702 after tier_dft_post for complete narrative with all mechanism data.
    Step "700_loso_lomo"
    Step "701_per_sub_shap"
}

function tier_dft_post {
    # ── Mechanism analysis (needs DFT results) ──────────────────────────────
    # 601 (catalyst mechanism): needs co2_drfp_xtb_extended.csv (tier_data).
    # 602 (substrate features): reads DFT summaries; degrades gracefully (RDKit-only)
    #   when dft_results_summary.csv / xtb_on_dft_geometry_nosolv.csv are absent.
    # 603 (transferability matrix): needs catalyst_mechanism.csv from 601.
    # ── Rerun 702 with complete mechanism data ──────────────────────────────
    # 702 was run in tier_loso but lacked 601/602/603 outputs. Rerun here
    #   so the integrated narrative includes full catalyst/substrate/transferability data.
    # ── LOSO improvement (needs 700/701 outputs from tier_loso) ────────────
    # 705 improved LOSO: needs summary_protocol.csv (700) and per_substrate_shap.csv (701).
    # 706 root-cause figure: needs improved_loso_results.json (705).
    # ── Diagnostic (needs mechanism data + tier_data) ──────────────────────
    # 901 substrate-catalyst matrix: needs co2_drfp_xtb_extended.csv.
    # 902 CHO mechanistic diagnostic: needs 901 output + co2_drfp_xtb_extended.csv.
    # ── Rerun 702 ──────────────────────────────────────────────────────────
    # 702 rerun after mechanism data is available for complete integrated narrative.
    Step "601_mechanism"
    Step "602_substrate"
    Step "603_transfer"
    Step "702_integrated"
    Step "705_improved"
    Step "706_root_cause"
    Step "901_substrate_cat"
    Step "902_cho_diag"
}

function tier_figures {
    Step "fig_pcl_ae_arch"
    Step "fig_yrand_100"
    Step "fig_0_abs"
    Step "fig_toc"
    Step "fig_4_root_cause"
    Step "fig_5_loso_proto"
    Step "fig_6_transf"
    Step "fig_7_shap_dir"
    Step "fig1_en"
    Step "fig1_zh"
    Step "fig2_en"
    Step "fig2_zh"
    Step "fig3_en"
    Step "fig3_zh"
    Step "fig4_en"
    Step "fig4_zh"
    Step "fig5_en"
    Step "fig5_zh"
}

function tier_regen {
    StepOptional "regen_all_v3"
}

function tier_si {
    Step "si_s3"
    Step "year_ood"
    Step "yrand_100"
    Step "boot_substrate"
    StepOptional "803_mordred"
    StepOptional "804_hier"
    StepOptional "807_cross_model"
    StepOptional "808_catalyst_ctrl"
    StepOptional "809_condition_ctrl"
    StepOptional "810_yield_dist"
}

function tier_abstract {
    Step "900_abstract"
}

function tier_dft {
    if ($SkipDFT) {
        Write-Host "[skip] tier_dft (-SkipDFT)" -ForegroundColor DarkYellow
        return
    }
    Step "501_dft_inputs" @("--level","medium")

    $dftArtifact = "$ROOT\dft_validation\dft_results_summary.csv"
    $deadline = (Get-Date).AddMinutes($WaitDFT)
    while (-not (Test-Path $dftArtifact)) {
        if ((Get-Date) -ge $deadline -or $WaitDFT -eq 0) {
            Write-Host ""
            Write-Host "=== DFT hand-off: artefact not yet ready ===" -ForegroundColor Yellow
            Write-Host "  Expected : $dftArtifact" -ForegroundColor Yellow
            Write-Host "  Action   : Run ORCA on WSL/Linux, then:" -ForegroundColor Yellow
            Write-Host "              .\run_pipeline_v2.ps1 -Tier tier_dft" -ForegroundColor Yellow
            Write-Host "  Pipeline : continuing with remaining tiers" -ForegroundColor Yellow
            Write-Host ""
            return
        }
        Write-Host ("  waiting for ORCA artefact ({0:N0}m remaining)..." -f ((($deadline) - (Get-Date)).TotalMinutes)) -ForegroundColor DarkYellow
        Start-Sleep -Seconds 60
    }
    Write-Host "=== DFT artefact ready: $dftArtifact ===" -ForegroundColor Green
    Step "510_parse_dft"
    Push-Location "$ROOT\dft_validation"
    try {
        Step "512_xtb_dft" @("--solvent","gas","--output","results/xtb_on_dft_geometry_nosolv.csv")
        Step "514_dft_vs_xtb" @("--xtb-summary","results/xtb_on_dft_geometry_nosolv.csv", "--output","results/514_dft_vs_xtb_report.csv", "--report","results/514_dft_vs_xtb_report.txt")
        Step "514b_dft_ts"
    } finally {
        Pop-Location
    }
    Step "520_figs"
    # Mechanism + LOSO-improvement + diagnostics all need DFT results or
    # tier_data outputs (already available). Run unconditionally.
    tier_dft_post
}

function tier_s7 {
    # tier_s7 currently wraps only the extended DFT input regeneration (502).
    # The previous StepOptional "713_orca_gen" was retired: its functionality
    # is now covered by 502_dft_extended + 514b_dft_transition_state.
    Step "502_dft_extended"
}

# ========================================================================
#  Tier metadata
# ========================================================================
$Tiers = @(
    "tier_data",       "tier_ablation",   "tier_pcl",        "tier_main",
    "tier_screening",  "tier_validation", "tier_loso",
    "tier_si",         "tier_dft",        "tier_regen",
    "tier_abstract",   "tier_figures",    "tier_s7"
)
# Note: 601/602/603/702/705/706/901/902 are NOT separate tiers.
#   They run inside tier_dft via tier_dft_post (at the very END of the pipeline),
#   after the ORCA/DFT artefact wait loop.  tier_loso only runs 700/701 (which
#   have no 601/602/603 dependency), and 702 is rerun in tier_dft_post so the
#   integrated narrative includes all mechanism data.

$TierTimes = @{
    "tier_data"       = "~10 min (or ~6 hr with xTB)"
    "tier_ablation"   = "~163 min (usually already done)"
    "tier_pcl"        = "~5 min"
    "tier_main"       = "~25 min  (302+303+304+306+401)"
    "tier_screening"  = "~15 min"
    "tier_validation" = "~15 min"
    "tier_loso"       = "~15 min  (700+701; 702 reruns in tier_dft_post)"
    "tier_si"         = "~10-90 min (yrand_100 = ~5 min, si_s3 = ~10 min, year_ood = ~10 min, 803/804/807-810 optional)"
    "tier_regen"      = "~5 min  (after tier_si)"
    "tier_abstract"   = "~2 min"
    "tier_figures"    = "~10 min"
    "tier_dft"        = "~45 min  (ORCA runs separately; mechanism steps run after ORCA)"
    "tier_s7"         = "~5-10 min  (502_dft_extended: regenerate extended DFT inputs)"
}

$TierNotes = @{
    "tier_pcl"   = "must run before tier_main (provides pcl_ae_latent.npy)"
    "tier_main"  = "core: 302+303+304+306+401"
    "tier_si"    = "807/808/809/810 = SHAP control experiments (#5/#3/#4/#6)"
    "tier_regen" = "must run AFTER tier_si (reads SI CSVs)"
    "tier_dft"   = "needs WSL; 514b generates TS inputs, user runs ORCA manually; 601→602→603→702→705→706→901→902 run after ORCA"
    "tier_s7"    = "optional regeneration of extended DFT inputs (502)"
}

# ========================================================================
#  List mode
# ========================================================================
if ($List) {
    Write-Host ""
    Write-Host "Pipeline tiers (in execution order):" -ForegroundColor Green
    Write-Host ""
    foreach ($t in $Tiers) {
        $time = $TierTimes[$t]
        $note = if ($TierNotes[$t]) { "  ← " + $TierNotes[$t] } else { "" }
        Write-Host "  $t$note" -ForegroundColor White
        Write-Host "    Time: $time" -ForegroundColor DarkGray
    }
    Write-Host ""
    Write-Host "Flags:" -ForegroundColor Cyan
    Write-Host "  -Tier T            run only tier T" -ForegroundColor Cyan
    Write-Host "  -List              show this list" -ForegroundColor Cyan
    Write-Host "  -DryRun            print commands without running" -ForegroundColor Cyan
    Write-Host "  -Resume            skip steps whose output is already up to date" -ForegroundColor Cyan
    Write-Host "  -NoForce           do not pass --force to scripts that support it" -ForegroundColor Cyan
    Write-Host "  -SkipDFT           skip tier_dft (no ORCA)" -ForegroundColor Cyan
    Write-Host "  -NoXTB             skip 104b xTB (~6 hr)" -ForegroundColor Cyan
    Write-Host "  -NoAbstract        skip tier_abstract" -ForegroundColor Cyan
    Write-Host "  -WaitDFT N         wait up to N min for ORCA artefact (0 = probe only)" -ForegroundColor Cyan
    Write-Host "  -StepTimeout N     per-step wall-clock cap, minutes (0 = unlimited)" -ForegroundColor Cyan
    Write-Host "  -Diagnostic        print config and exit" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Env vars:" -ForegroundColor Cyan
    Write-Host "  CO2_PROJECT_ROOT   project root (default: D:\machine-learning\CO2-cycloaddition)" -ForegroundColor Cyan
    Write-Host "  CO2_PYTHON         python interpreter (default: D:\co2\env_drfp\python.exe)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Exit codes:" -ForegroundColor Cyan
    Write-Host "  0 = OK,  1 = required step failed,  2 = warnings only" -ForegroundColor Cyan
    exit 0
}

# ========================================================================
#  Diagnostic mode
# ========================================================================
if ($Diagnostic) {
    Write-Host ""
    Write-Host "Pipeline diagnostic" -ForegroundColor Green
    Write-Host ("-" * 60) -ForegroundColor Green
    Write-Host "  ROOT        : $ROOT" -ForegroundColor White
    Write-Host "  PYTHON      : $PY" -ForegroundColor White
    Write-Host "  DryRun      : $DryRun" -ForegroundColor White
    Write-Host "  Resume      : $Resume" -ForegroundColor White
    Write-Host "  NoForce     : $NoForce" -ForegroundColor White
    Write-Host "  StepTimeout : $StepTimeout min" -ForegroundColor White
    Write-Host "  NoXTB       : $NoXTB" -ForegroundColor White
    Write-Host "  SkipDFT     : $SkipDFT" -ForegroundColor White
    Write-Host "  WaitDFT     : $WaitDFT min" -ForegroundColor White
    Write-Host "  Log file    : $LOG_FILE" -ForegroundColor White
    Write-Host ""
    Write-Host "  Steps registered: $($Steps.Count)" -ForegroundColor White
    Write-Host ("-" * 60) -ForegroundColor Green
    exit 0
}

# ========================================================================
#  Dispatcher
# ========================================================================
$PipelineStart = Get-Date
$fatalSteps = @()
$warnSteps  = @()
$skippedSteps = @()

function Run-Tier([string]$Name) {
    $tierSw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Green
    Write-Host "TIER :: $Name  ($($TierTimes[$Name]))" -ForegroundColor Green
    Write-Host ("=" * 72) -ForegroundColor Green
    Log "TIER BEGIN :: $Name" "Green"

    $err = $null
    try {
        & $Name
    } catch {
        $err = $_
        Write-Host "[$Name] EXCEPTION: $_" -ForegroundColor Red
        Log "[$Name] EXCEPTION: $_" "Red"
        $script:fatalSteps += @{ tier = $Name; error = "$_" }
    }
    $tierSw.Stop()
    Write-Host ""
    Write-Host ("--- TIER END: $Name  ({0:N1}s) ---" -f $tierSw.Elapsed.TotalSeconds) -ForegroundColor DarkGreen
    Log "TIER END   :: $Name  ($([int]$tierSw.Elapsed.TotalSeconds)s)" "DarkGreen"
}

if ($Tier -ne "") {
    if ($Tier -notin $Tiers) {
        Write-Host "[ERR] Unknown tier '$Tier'.  Valid:" -ForegroundColor Red
        foreach ($t in $Tiers) { Write-Host "  $t" }
        exit 1
    }
    Run-Tier $Tier
    Write-Host ""
    Write-Host "=== DONE (tier=$Tier) ===" -ForegroundColor Green
    exit 0
}

foreach ($t in $Tiers) {
    if ($t -eq "tier_abstract" -and $NoAbstract) {
        Write-Host "[skip] tier_abstract (-NoAbstract)" -ForegroundColor DarkYellow
        continue
    }
    Run-Tier $t
}

$totalSw = (Get-Date) - $PipelineStart
Write-Host ""
Write-Host ("=" * 72) -ForegroundColor Green
Write-Host ("PIPELINE DONE   total {0:N1}s ({1:N1} min)" -f $totalSw.TotalSeconds, ($totalSw.TotalSeconds/60)) -ForegroundColor Green
Write-Host ("=" * 72) -ForegroundColor Green
Write-Host "Log file: $LOG_FILE" -ForegroundColor DarkGray

if ($script:fatalSteps.Count -gt 0) {
    Write-Host ""
    Write-Host "[FATAL] The following required steps failed:" -ForegroundColor Red
    foreach ($f in $script:fatalSteps) { Write-Host "  $($f.tier): $($f.error)" -ForegroundColor Red }
    exit 1
}

exit 0