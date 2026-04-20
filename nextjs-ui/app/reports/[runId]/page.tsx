'use client';

import Link from 'next/link';
import { notFound, useParams } from 'next/navigation';
import { useCampaign } from '@/lib/campaign-context';
import { formatCategoryLabel, weaknessBadgeClass } from '@/lib/campaign';
import { ModelBadge } from '@/components/model-badge';

export default function RunReportPage() {
  const params = useParams<{ runId: string }>();
  const runId = decodeURIComponent(params.runId);
  const { getCampaignByRunId } = useCampaign();
  const campaignResult = getCampaignByRunId(runId);

  if (!campaignResult) {
    notFound();
  }

  const reports = campaignResult.validation_reports?.length
    ? [...campaignResult.validation_reports].reverse()
    : [];

  return (
    <>
      <header className="sticky top-0 z-10 border-b border-[var(--admin-border)] bg-[var(--admin-card)] shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-3 md:px-6">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <Link href="/" className="hover:text-[var(--admin-primary)]">
                Home
              </Link>
              <span className="text-slate-300">/</span>
              <Link href="/reports" className="hover:text-[var(--admin-primary)]">
                Run reports
              </Link>
              <span className="text-slate-300">/</span>
              <span className="font-medium text-[var(--admin-primary)]">Full report</span>
            </div>
            <h1 className="mt-0.5 text-lg font-semibold text-slate-900 md:text-xl">
              {campaignResult.target_model ?? 'Unknown target'} full report
            </h1>
            <div className="mt-2">
              <ModelBadge modelId={campaignResult.target_model} />
            </div>
            <p className="mt-1 font-mono text-[11px] text-slate-400">
              Run {campaignResult.campaign_meta?.run_id ?? runId} · completed{' '}
              {campaignResult.campaign_meta?.completed_at ?? 'unknown'} ·{' '}
              {campaignResult.campaign_meta?.algorithm ?? 'unknown'} ·{' '}
              {campaignResult.campaign_meta?.executor ?? 'unknown'}
            </p>
          </div>
          <Link
            href="/reports"
            className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            Back to runs
          </Link>
        </div>
      </header>

      <main className="flex-1 px-4 py-6 md:px-6">
        <div className="mb-4 text-sm text-slate-600">
          Showing <strong>{reports.length}</strong> prompt report{reports.length === 1 ? '' : 's'} for this run.
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          {reports.map((r) => (
            <article
              key={`${campaignResult.campaign_meta?.run_id ?? runId}-${r.test_id}`}
              className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm"
            >
              <div className="flex flex-wrap items-start justify-between gap-2 border-b border-slate-100 pb-3">
                <div className="min-w-0">
                  <p className="text-[11px] font-mono text-slate-400">{r.prompt_id}</p>
                  <h2 className="mt-1 text-base font-semibold text-slate-900">{r.prompt_title}</h2>
                </div>
                <span
                  className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold ${
                    r.verdict.includes('Weakness')
                      ? 'bg-amber-100 text-amber-900'
                      : 'bg-slate-100 text-slate-700'
                  }`}
                >
                  #{r.test_id} · {r.verdict}
                </span>
              </div>

              <p className="mt-3 text-sm leading-relaxed text-slate-600">{r.prompt_excerpt}</p>

              <div className="mt-3 flex flex-wrap gap-2">
                <span
                  className={`inline-flex rounded border px-2 py-0.5 text-xs font-medium ${weaknessBadgeClass(r.weakness_type)}`}
                >
                  {formatCategoryLabel(r.weakness_type)}
                </span>
                <span className="inline-flex rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-600">
                  {r.corpus_source}
                </span>
              </div>

              <div className="mt-4 rounded-lg bg-slate-50 p-4 text-sm">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  How this prompt was validated
                </h3>
                <p className="mt-2 text-slate-700">{r.validation_method}</p>
                <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
                  <div>
                    <dt className="text-xs text-slate-500">Composite score</dt>
                    <dd className="mt-1">
                      <span
                        className={`inline-flex items-center rounded-md px-2 py-1 text-sm font-bold ${
                          r.scoring.composite_score >= 0.7
                            ? 'bg-red-500 text-white'
                            : r.scoring.composite_score >= 0.5
                            ? 'bg-amber-500 text-white'
                            : 'bg-emerald-500 text-white'
                        }`}
                      >
                        {r.scoring.composite_score.toFixed(3)}
                      </span>
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-500">Bandit reward</dt>
                    <dd className="font-mono font-semibold text-slate-900">{r.scoring.bandit_reward}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-500">UCB at selection</dt>
                    <dd className="font-mono font-semibold text-slate-900">
                      {r.scoring.ucb_at_selection ?? '—'}
                    </dd>
                  </div>
                </dl>
              </div>
            </article>
          ))}
        </div>
      </main>
    </>
  );
}
