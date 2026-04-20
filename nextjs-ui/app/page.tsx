'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import {
  Play,
  TrendingUp,
  Target,
  Award,
  Activity,
  Brain,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from 'recharts';
import axios from 'axios';
import { useCampaign } from '@/lib/campaign-context';
import { ModelBadge } from '@/components/model-badge';
import {
  TARGET_MODEL_OPTIONS,
  getClientBackendUrl,
  formatCategoryLabel,
  weaknessBadgeClass,
  type CampaignResult,
} from '@/lib/campaign';

export default function Home() {
  const { campaignResult, setCampaignResult } = useCampaign();
  const result = campaignResult;

  const [algorithm, setAlgorithm] = useState('ucb1');
  const [testCount, setTestCount] = useState(50);
  const [targetModel, setTargetModel] = useState<string>(TARGET_MODEL_OPTIONS[0].id);
  const [isLoading, setIsLoading] = useState(false);
  const [requestStartedAt, setRequestStartedAt] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [runError, setRunError] = useState<string | null>(null);

  const selectedTarget = TARGET_MODEL_OPTIONS.find((m) => m.id === targetModel);

  useEffect(() => {
    if (!isLoading || requestStartedAt == null) return;

    const interval = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - requestStartedAt) / 1000));
    }, 1000);

    return () => window.clearInterval(interval);
  }, [isLoading, requestStartedAt]);

  const runCampaign = async () => {
    setIsLoading(true);
    setRunError(null);
    setRequestStartedAt(Date.now());
    setElapsedSeconds(0);

    try {
      const response = await axios.post<CampaignResult>(`${getClientBackendUrl()}/api/run_campaign`, {
        algorithm,
        test_count: testCount,
        target_model: targetModel,
      }, {
        timeout: 0,
      });
      setCampaignResult(response.data);
    } catch (error) {
      console.error('Error running campaign:', error);
      if (axios.isAxiosError(error)) {
        const detail =
          (typeof error.response?.data === 'object' &&
            error.response?.data &&
            'detail' in error.response.data &&
            typeof error.response.data.detail === 'string' &&
            error.response.data.detail) ||
          error.message;
        setRunError(detail);
      } else {
        setRunError('Failed to run campaign.');
      }
    } finally {
      setIsLoading(false);
      setRequestStartedAt(null);
    }
  };

  const prepareArmData = () => {
    if (!result?.arm_statistics) return [];
    return Object.entries(result.arm_statistics).map(([name, stats]) => ({
      name: formatCategoryLabel(name),
      pulls: stats.pulls,
      mean_reward: stats.mean_reward,
      success_rate: stats.success_rate * 100,
      // Add color coding based on mean reward
      reward_color: stats.mean_reward >= 0.6 ? '#10b981' : stats.mean_reward >= 0.4 ? '#f59e0b' : '#ef4444',
    }));
  };

  const prepareLearningData = () => {
    if (!result?.learning_progress) return [];
    const { cumulative_rewards, success_rates } = result.learning_progress;
    return cumulative_rewards.map((reward, i) => ({
      test: i + 1,
      reward,
      success_rate: success_rates[i],
    }));
  };

  const prepareUCBData = () => {
    if (!result?.ucb_values) return [];
    return Object.entries(result.ucb_values)
      .filter(([, value]) => value !== Number.POSITIVE_INFINITY && Number.isFinite(value))
      .map(([name, value]) => ({
        name: formatCategoryLabel(name),
        value,
      }));
  };

  return (
    <>
      <header className="sticky top-0 z-10 border-b border-[var(--admin-border)] bg-[var(--admin-card)] shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-3 md:px-6">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span>Home</span>
              <span className="text-slate-300">/</span>
              <span className="font-medium text-slate-700">Validation</span>
              <span className="text-slate-300">/</span>
              <span className="font-medium text-[var(--admin-primary)]">Campaign</span>
            </div>
            <h1 className="mt-0.5 truncate text-lg font-semibold text-slate-900 md:text-xl">
              RL validation dashboard
            </h1>
            <p className="mt-1 text-sm text-slate-500">
              Live adversarial campaigns with saved backend history and reopenable full reports.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex flex-col gap-1 rounded-lg border border-[var(--admin-border)] bg-slate-50/80 px-3 py-2">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                Target model
              </span>
              <ModelBadge modelId={targetModel} compact />
              <select
                value={targetModel}
                onChange={(e) => setTargetModel(e.target.value)}
                className="max-w-[220px] border-0 bg-transparent text-sm font-semibold text-slate-800 outline-none focus:ring-0"
              >
                {TARGET_MODEL_OPTIONS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
              {selectedTarget && (
                <span className="text-[11px] text-slate-500">{selectedTarget.family}</span>
              )}
            </div>
            {result?.campaign_meta && (
              <div className="hidden rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-800 sm:block">
                Last run: {result.target_model ?? result.campaign_meta.algorithm} · {result.campaign_meta.completed_at.slice(0, 16).replace('T', ' ')}
              </div>
            )}
            <Link
              href="/reports"
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50"
            >
              View reports
              {result?.validation_reports?.length ? (
                <span className="ml-1.5 rounded-full bg-[var(--admin-primary)] px-1.5 py-0.5 text-[10px] text-white">
                  {result.validation_reports.length}
                </span>
              ) : null}
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1 px-4 py-6 md:px-6">
        <div className="mb-6 overflow-hidden rounded-[28px] border border-slate-200 bg-[radial-gradient(circle_at_top_left,_rgba(54,92,245,0.16),_transparent_40%),linear-gradient(135deg,_#ffffff_0%,_#eef4ff_100%)] p-6 shadow-sm">
          <div className="max-w-2xl">
              <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-[var(--admin-primary)]">
                Validation Studio
              </p>
              <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">
                Persistent runs, polished reports, and clearer target model identity
              </h2>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                Completed campaigns are now persisted on the backend, mirrored into the interface, and grouped into
                full report views you can reopen later.
              </p>
          </div>
        </div>

        <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-12">
          <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm md:col-span-4">
            <h3 className="mb-3 flex items-center text-sm font-semibold text-slate-800">
              <Target className="mr-2 h-4 w-4 text-[var(--admin-primary)]" />
              Bandit algorithm
            </h3>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none ring-[var(--admin-primary)] focus:ring-2"
            >
              <option value="ucb1">UCB1 (standard)</option>
              <option value="ucb1_tuned">UCB1-Tuned (variance-aware)</option>
              <option value="ucbv">UCB-V (variance-based)</option>
            </select>
            <p className="mt-2 text-xs text-slate-500">Exploration vs. exploitation for test arms.</p>
          </div>

          <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm md:col-span-4">
            <h3 className="mb-3 flex items-center text-sm font-semibold text-slate-800">
              <Activity className="mr-2 h-4 w-4 text-[var(--admin-primary)]" />
              Test budget
            </h3>
            <input
              type="number"
              value={testCount}
              onChange={(e) => setTestCount(parseInt(e.target.value, 10) || 10)}
              min={10}
              max={500}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none ring-[var(--admin-primary)] focus:ring-2"
            />
            <p className="mt-2 text-xs text-slate-500">Iterations per campaign (10–500).</p>
          </div>

          <div className="flex items-stretch md:col-span-4">
            <button
              type="button"
              onClick={runCampaign}
              disabled={isLoading}
              className={`flex w-full items-center justify-center rounded-xl px-6 text-sm font-semibold text-white shadow-md transition ${
                isLoading
                  ? 'cursor-not-allowed bg-slate-400'
                  : 'bg-[var(--admin-primary)] hover:bg-[var(--admin-primary-hover)] active:scale-[0.99]'
              }`}
            >
              {isLoading ? (
                <span className="flex flex-col items-center gap-2">
                  <svg
                    className="h-5 w-5 animate-spin"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  <span className="text-xs">Running…</span>
                  <span className="mt-1 text-[10px] text-white/90">
                    Waiting for backend response | {elapsedSeconds}s elapsed
                  </span>
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Play className="h-5 w-5" />
                  Run campaign
                </span>
              )}
            </button>
          </div>
        </div>

        {isLoading && (
          <div className="mb-6 rounded-xl border border-blue-200 bg-blue-50 p-4 shadow-sm">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-blue-900">Campaign in progress</h2>
                <p className="mt-1 text-sm text-blue-800">
                  Waiting for the backend to finish processing before the dashboard updates.
                </p>
              </div>
              <div className="text-right text-xs font-medium text-blue-700">
                <div>{testCount} tests requested</div>
                <div>{elapsedSeconds}s elapsed</div>
              </div>
            </div>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-blue-100">
              <div className="h-full w-1/3 animate-pulse rounded-full bg-blue-500" />
            </div>
          </div>
        )}

        {runError && (
          <div className="mb-6 rounded-xl border border-red-200 bg-red-50 p-4 shadow-sm">
            <h2 className="text-sm font-semibold text-red-900">Campaign failed to start</h2>
            <p className="mt-1 text-sm text-red-800">{runError}</p>
          </div>
        )}

        {!result && !isLoading && (
          <div className="rounded-xl border border-dashed border-slate-200 bg-white/60 py-16 text-center shadow-sm">
            <Brain className="mx-auto mb-4 h-14 w-14 text-slate-300" />
            <h2 className="text-lg font-semibold text-slate-700">Ready to validate</h2>
            <p className="mx-auto mt-2 max-w-md text-sm text-slate-500">
              Choose an open-weight <strong>target model</strong> above, set the algorithm and budget, then
              run a campaign. Open <strong>Prompt reports</strong> in the sidebar (or{' '}
              <Link href="/reports" className="font-medium text-[var(--admin-primary)] hover:underline">
                View reports
              </Link>
              ) for full per-prompt validation and scores.
            </p>
          </div>
        )}

        {result && (
          <>
            <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
                <div className="mb-1 flex items-center text-xs font-medium text-slate-500">
                  <Activity className="mr-1.5 h-3.5 w-3.5" /> Total tests
                </div>
                <div className="text-2xl font-bold text-slate-900">{result.total_tests}</div>
              </div>
              <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
                <div className="mb-1 flex items-center text-xs font-medium text-slate-500">
                  <CheckCircle className="mr-1.5 h-3.5 w-3.5 text-emerald-600" /> Weakness signals
                </div>
                <div className="text-2xl font-bold text-emerald-700">{result.weaknesses_found}</div>
              </div>
              <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
                <div className="mb-1 flex items-center text-xs font-medium text-slate-500">
                  <Award className="mr-1.5 h-3.5 w-3.5 text-amber-600" /> Success rate
                </div>
                <div className="text-2xl font-bold text-slate-900">{result.success_rate.toFixed(1)}%</div>
              </div>
              <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
                <div className="mb-1 flex items-center text-xs font-medium text-slate-500">
                  <TrendingUp className="mr-1.5 h-3.5 w-3.5 text-[var(--admin-primary)]" /> Cumulative reward
                </div>
                <div className="text-2xl font-bold text-slate-900">
                  {result.cumulative_reward.toFixed(2)}
                </div>
              </div>
            </div>

            <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
              <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
                <h3 className="mb-4 flex items-center text-sm font-semibold text-slate-800">
                  <Target className="mr-2 h-4 w-4 text-[var(--admin-primary)]" />
                  Weakness type performance
                </h3>
                <p className="mb-4 text-xs text-slate-500">
                  Bars show tests run per weakness. Line shows success rate (weakness found). 
                  <span className="ml-2 font-medium text-emerald-600">Green ≥60%</span> (effective),{' '}
                  <span className="font-medium text-amber-600">Amber 40–59%</span> (moderate),{' '}
                  <span className="font-medium text-red-600">Red &lt;40%</span> (needs attention).
                </p>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={prepareArmData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="name"
                      angle={-35}
                      textAnchor="end"
                      stroke="#94a3b8"
                      tick={{ fontSize: 10 }}
                      height={70}
                    />
                    <YAxis
                      yAxisId="left"
                      stroke="#365cf5"
                      tick={{ fontSize: 11 }}
                      label={{
                        value: 'Tests',
                        angle: -90,
                        position: 'insideLeft',
                        style: { fill: '#64748b', fontSize: 11 },
                      }}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      stroke="#10b981"
                      tick={{ fontSize: 11 }}
                      label={{
                        value: 'Success %',
                        angle: 90,
                        position: 'insideRight',
                        style: { fill: '#64748b', fontSize: 11 },
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#fff',
                        border: '1px solid #e2e8f0',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                    <Bar
                      yAxisId="left"
                      dataKey="pulls"
                      name="Tests run"
                      radius={[4, 4, 0, 0]}
                      fill="#8884d8"
                      shape={(props: any) => {
                        const { x, y, width, height, payload } = props;
                        return <rect x={x} y={y} width={width} height={height} fill={payload.reward_color} rx={4} ry={4} />;
                      }}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="success_rate"
                      stroke="#10b981"
                      strokeWidth={2}
                      name="Success rate %"
                      dot={(props: any) => {
                        const { cx, cy, value } = props;
                        let fillColor = '#ef4444'; // red (low)
                        if (value >= 60) {
                          fillColor = '#10b981'; // emerald (high)
                        } else if (value >= 40) {
                          fillColor = '#f59e0b'; // amber (moderate)
                        }
                        return <circle cx={cx} cy={cy} r={4} fill={fillColor} />;
                      }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
                <h3 className="mb-4 flex items-center text-sm font-semibold text-slate-800">
                  <Brain className="mr-2 h-4 w-4 text-[var(--admin-primary)]" />
                  Learning progress
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={prepareLearningData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="test"
                      stroke="#94a3b8"
                      tick={{ fontSize: 11 }}
                      label={{
                        value: 'Test #',
                        position: 'insideBottom',
                        offset: -4,
                        style: { fill: '#64748b', fontSize: 11 },
                      }}
                    />
                    <YAxis
                      yAxisId="left"
                      stroke="#d97706"
                      tick={{ fontSize: 11 }}
                      label={{
                        value: 'Reward',
                        angle: -90,
                        position: 'insideLeft',
                        style: { fill: '#64748b', fontSize: 11 },
                      }}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      stroke="#7c3aed"
                      tick={{ fontSize: 11 }}
                      label={{
                        value: 'Success',
                        angle: 90,
                        position: 'insideRight',
                        style: { fill: '#64748b', fontSize: 11 },
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#fff',
                        border: '1px solid #e2e8f0',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="reward"
                      stroke="#d97706"
                      strokeWidth={2}
                      name="Cumulative reward"
                      dot={false}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="success_rate"
                      stroke="#7c3aed"
                      strokeWidth={2}
                      name="Running success"
                      dot={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="mb-6 rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
              <h3 className="mb-4 flex items-center text-sm font-semibold text-slate-800">
                <Activity className="mr-2 h-4 w-4 text-[var(--admin-primary)]" />
                UCB values (finite arms)
              </h3>
              <p className="mb-4 text-xs text-slate-500">
                UCB = exploitation (mean reward) + exploration bonus. Higher values indicate arms that need more testing.
                <span className="ml-2 font-medium text-red-600">Red ≥1.0</span> (high priority),{' '}
                <span className="font-medium text-amber-600">Amber 0.7–0.99</span> (moderate),{' '}
                <span className="font-medium text-emerald-600">Emerald &lt;0.7</span> (low priority).
              </p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={prepareUCBData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="name"
                    angle={-35}
                    textAnchor="end"
                    stroke="#94a3b8"
                    tick={{ fontSize: 10 }}
                    height={72}
                  />
                  <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Bar
                    dataKey="value"
                    name="UCB"
                    radius={[4, 4, 0, 0]}
                    fill="#8884d8"
                    shape={(props: any) => {
                      const { x, y, width, height, value } = props;
                      let fillColor = '#10b981'; // emerald (low)
                      if (value >= 1.0) {
                        fillColor = '#ef4444'; // red (high)
                      } else if (value >= 0.7) {
                        fillColor = '#f59e0b'; // amber (moderate)
                      }
                      return <rect x={x} y={y} width={width} height={height} fill={fillColor} rx={4} ry={4} />;
                    }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mb-6 rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
              <h3 className="mb-4 flex items-center text-sm font-semibold text-slate-800">
                <Award className="mr-2 h-4 w-4 text-amber-600" />
                Recommendations
              </h3>
              <div className="space-y-3">
                {result.recommendations.map((rec, idx) => (
                  <div
                    key={idx}
                    className={`rounded-lg border border-slate-100 bg-slate-50/80 p-4 border-l-4 ${
                      rec.priority === 'high'
                        ? 'border-l-red-500'
                        : rec.priority === 'medium'
                          ? 'border-l-amber-500'
                          : 'border-l-emerald-500'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <span className="text-xl">{rec.icon}</span>
                      <div>
                        <h4 className="font-semibold text-slate-900">{rec.title}</h4>
                        <p className="mt-1 text-sm text-slate-600">{rec.description}</p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <span className="rounded bg-white px-2 py-0.5 text-xs font-medium text-slate-600 ring-1 ring-slate-200">
                            Focus: {rec.focus_area}
                          </span>
                          <span
                            className={`rounded px-2 py-0.5 text-xs font-medium text-white ${
                              rec.confidence > 0.7 ? 'bg-emerald-600' : 'bg-amber-600'
                            }`}
                          >
                            Confidence {(rec.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-[var(--admin-border)] bg-[var(--admin-card)] p-5 shadow-sm">
              <h3 className="mb-1 flex items-center text-sm font-semibold text-slate-800">
                <Activity className="mr-2 h-4 w-4 text-[var(--admin-primary)]" />
                Recent test results
              </h3>
              <p className="mb-4 text-xs text-slate-500">
                Final five steps of this run only (same data as Prompt reports). Demo scores are deterministic from
                prompt + weakness + step + model — not unrelated RNG.
                {result.campaign_meta && (
                  <span className="ml-1 font-mono text-slate-400">
                    Run {result.campaign_meta.run_id.slice(0, 8)}…
                  </span>
                )}
              </p>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                      <th className="pb-3 pr-3 font-medium">#</th>
                      <th className="pb-3 pr-3 font-medium">Prompt</th>
                      <th className="pb-3 pr-3 font-medium">Weakness</th>
                      <th className="pb-3 pr-3 font-medium">Result</th>
                      <th className="pb-3 pr-3 font-medium">Confidence</th>
                      <th className="pb-3 pr-3 font-medium">Score</th>
                      <th className="pb-3 font-medium">UCB</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.recent_tests
                      .slice()
                      .reverse()
                      .map((test) => (
                        <tr
                          key={test.test_id}
                          className="border-b border-slate-100 hover:bg-slate-50/80"
                        >
                          <td className="py-3 pr-3 text-slate-500">{test.test_id}</td>
                          <td
                            className="max-w-[200px] truncate py-3 pr-3 text-slate-700"
                            title={test.prompt_title ?? ''}
                          >
                            {test.prompt_title ?? '—'}
                          </td>
                          <td className="py-3 pr-3">
                            <span
                              className={`inline-flex rounded border px-2 py-0.5 text-xs font-medium ${weaknessBadgeClass(test.weakness_type)}`}
                            >
                              {formatCategoryLabel(test.weakness_type)}
                            </span>
                          </td>
                          <td className="py-3 pr-3">
                            {test.success ? (
                              <span className="flex items-center font-medium text-emerald-700">
                                <CheckCircle className="mr-1 h-4 w-4" /> Signal
                              </span>
                            ) : (
                              <span className="flex items-center font-medium text-slate-500">
                                <XCircle className="mr-1 h-4 w-4" /> None
                              </span>
                            )}
                          </td>
                          <td className="py-3 pr-3">
                            <span
                              className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-semibold ${
                                test.confidence >= 0.7
                                  ? 'bg-red-100 text-red-800 border border-red-200'
                                  : test.confidence >= 0.5
                                  ? 'bg-amber-100 text-amber-800 border border-amber-200'
                                  : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                              }`}
                            >
                              {(test.confidence * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="py-3 pr-3">
                            <span
                              className={`inline-flex items-center rounded-md px-2 py-1 text-sm font-bold ${
                                test.composite_score != null && test.composite_score >= 0.7
                                  ? 'bg-red-500 text-white'
                                  : test.composite_score != null && test.composite_score >= 0.5
                                  ? 'bg-amber-500 text-white'
                                  : 'bg-emerald-500 text-white'
                              }`}
                            >
                              {test.composite_score != null ? test.composite_score.toFixed(3) : '—'}
                            </span>
                          </td>
                          <td className="py-3 font-mono">
                            <span
                              className={`inline-flex items-center rounded-md px-2 py-1 text-xs font-bold ${
                                test.ucb_value >= 1.0
                                  ? 'bg-red-500 text-white'
                                  : test.ucb_value >= 0.7
                                  ? 'bg-amber-500 text-white'
                                  : 'bg-emerald-500 text-white'
                              }`}
                            >
                              {test.ucb_value.toFixed(3)}
                            </span>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>

      <footer className="border-t border-[var(--admin-border)] bg-[var(--admin-card)] px-6 py-4">
        <p className="text-center text-xs text-slate-500">
          Popper RL validation agent · UCB multi-armed bandits ·{' '}
          <a
            href="https://github.com/Humanitariansai/Popper"
            className="text-[var(--admin-primary)] hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            Popper framework
          </a>
        </p>
      </footer>
    </>
  );
}
