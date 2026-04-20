'use client';

import Link from 'next/link';
import { FileText } from 'lucide-react';
import { useCampaign } from '@/lib/campaign-context';
import { ModelBadge } from '@/components/model-badge';

export default function ReportsPage() {
  const { campaignHistory } = useCampaign();

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
              <span className="font-medium text-[var(--admin-primary)]">Run reports</span>
            </div>
            <h1 className="mt-0.5 text-lg font-semibold text-slate-900 md:text-xl">
              Historical campaign runs
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-slate-500">
              Successful campaigns are stored in your browser so you can reopen older prompt reports instead of losing
              them on the next run.
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-right text-sm">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
              Stored runs
            </div>
            <div className="font-semibold text-slate-800">{campaignHistory.length}</div>
          </div>
        </div>
      </header>

      <main className="flex-1 px-4 py-6 md:px-6">
        {campaignHistory.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-200 bg-white/80 py-16 text-center shadow-sm">
            <FileText className="mx-auto mb-4 h-14 w-14 text-slate-300" />
            <h2 className="text-lg font-semibold text-slate-700">No saved runs yet</h2>
            <p className="mx-auto mt-2 max-w-md text-sm text-slate-500">
              Run a campaign from the{' '}
              <Link href="/" className="font-medium text-[var(--admin-primary)] hover:underline">
                dashboard
              </Link>{' '}
              and it will show up here with a full prompt report view.
            </p>
          </div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            {campaignHistory.map((run) => {
              const runId = run.campaign_meta?.run_id ?? 'unknown-run';
              return (
                <article
                  key={runId}
                  className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-[11px] font-mono text-slate-400">{runId}</p>
                      <h2 className="mt-1 text-base font-semibold text-slate-900">
                        {run.target_model ?? 'Unknown target'}
                      </h2>
                      <div className="mt-2">
                        <ModelBadge modelId={run.target_model} compact />
                      </div>
                      <p className="mt-1 text-sm text-slate-500">
                        {run.campaign_meta?.completed_at ?? 'Unknown completion time'}
                      </p>
                    </div>
                    <Link
                      href={`/reports/${encodeURIComponent(runId)}`}
                      className="rounded-lg bg-[var(--admin-primary)] px-3 py-2 text-sm font-medium text-white hover:bg-[var(--admin-primary-hover)]"
                    >
                      Full report
                    </Link>
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                    <div className="rounded-lg bg-slate-50 p-3">
                      <div className="text-[10px] uppercase tracking-wide text-slate-500">Tests</div>
                      <div className="mt-1 font-semibold text-slate-900">{run.total_tests}</div>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3">
                      <div className="text-[10px] uppercase tracking-wide text-slate-500">Signals</div>
                      <div className="mt-1 font-semibold text-emerald-700">{run.weaknesses_found}</div>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3">
                      <div className="text-[10px] uppercase tracking-wide text-slate-500">Success</div>
                      <div className="mt-1 font-semibold text-slate-900">{run.success_rate.toFixed(1)}%</div>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3">
                      <div className="text-[10px] uppercase tracking-wide text-slate-500">Reports</div>
                      <div className="mt-1 font-semibold text-slate-900">{run.validation_reports?.length ?? 0}</div>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </main>
    </>
  );
}
