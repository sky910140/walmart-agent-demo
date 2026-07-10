# Recorded Demo Outputs

These outputs were generated locally on 2026-07-10 after downloading one latest 10-K for each of the ten configured companies, indexing 4,482 chunks, and downloading CSI 300 data through 2026-07-10. Model keys were intentionally absent, so the application disclosed its offline extractive mode.

## Twenty-year CSI 300 snapshot

Command:

```powershell
python -m finagent market --file data/market/csi300.csv --start 2006-07-10 --end 2026-07-10
```

Output:

```text
| Symbol | Start | End | Start close | End close | Change | Avg volume |
| sh000300 | 2006-07-10 | 2026-07-10 | 1412.12 | 4780.79 | 238.55% | 117740769 |
Source: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
```

The source sidecar reported 5,164 rows covering 2005-04-08 through 2026-07-10.

## Apple liquidity and debt risk

Command:

```powershell
python -m finagent ask "Summarize liquidity and debt-related risks." --company Apple --user demo-reviewer --trace
```

Selected output:

```text
Offline extractive mode is active because one or both required model credentials are unavailable.

- The value and liquidity of the Company's cash, cash equivalents and marketable securities may fluctuate substantially. [S1]
- Adverse economic conditions can lead to limitations on the Company's ability to issue new debt and reduced liquidity. [S2]
- Apple stated that cash, cash equivalents and marketable securities totaled $132.4 billion as of September 27, 2025, and described ongoing operating cash generation and access to debt markets as sufficient for its stated cash requirements. [S3]
```

`[S1]` to `[S3]` each pointed to the same original filing with distinct retrieved chunk IDs: Apple Inc. Form 10-K, filed 2025-10-31, accession `0000320193-25-000079`, [SEC archive](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm).

The trace reported DeepSeek V4 as planner/verifier and `doubao-seed-evolving` as analyst, all `remote=False` because no credentials were supplied. With both configured, the same evidence is sent to the two-model loop and remote output is accepted only when it retains valid `[S#]` labels.
