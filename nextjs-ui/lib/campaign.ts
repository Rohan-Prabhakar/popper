export interface ArmStats {
  pulls: number;
  mean_reward: number;
  success_rate: number;
  variance: number;
}

export interface LearningProgress {
  cumulative_rewards: number[];
  success_rates: number[];
}

export interface ValidationReport {
  test_id: number;
  prompt_id: string;
  prompt_title: string;
  prompt_excerpt: string;
  weakness_type: string;
  validation_method: string;
  scoring: {
    bandit_reward: number;
    ucb_at_selection: number | null;
    composite_score: number;
  };
  verdict: string;
  corpus_source: string;
}

export interface Recommendation {
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  icon: string;
  focus_area: string;
  confidence: number;
}

export interface RecentTest {
  test_id: number;
  weakness_type: string;
  success: boolean;
  confidence: number;
  reward: number;
  ucb_value: number;
  prompt_title?: string | null;
  composite_score?: number | null;
}

export interface CampaignMeta {
  run_id: string;
  started_at: string;
  completed_at: string;
  algorithm: string;
  num_tests: number;
  executor: string;
}

export interface CampaignResult {
  total_tests: number;
  weaknesses_found: number;
  success_rate: number;
  cumulative_reward: number;
  arm_statistics: Record<string, ArmStats>;
  learning_progress: LearningProgress;
  ucb_values: Record<string, number>;
  recommendations: Recommendation[];
  recent_tests: RecentTest[];
  target_model?: string;
  validation_reports?: ValidationReport[];
  campaign_meta?: CampaignMeta;
}

export const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8000';

export function getClientBackendUrl() {
  const configured = process.env.NEXT_PUBLIC_POPPER_BACKEND_URL;
  return (configured || DEFAULT_BACKEND_URL).replace(/\/+$/, '');
}

export type ModelOption = {
  id: string;
  label: string;
  family: string;
  logoText: string;
  accent: string;
};

export const TARGET_MODEL_OPTIONS = [
  { id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 70B Versatile', family: 'Groq', logoText: 'L3', accent: '#365cf5' },
  { id: 'qwen/qwen3-32b', label: 'Qwen 3 32B', family: 'Groq', logoText: 'Q3', accent: '#0f766e' },
  { id: 'llama-3.1-8b-instant', label: 'Llama 3.1 8B Instant', family: 'Groq', logoText: 'L1', accent: '#7c3aed' },
  { id: 'openai/gpt-oss-20b', label: 'GPT OSS 20B', family: 'Groq', logoText: 'O2', accent: '#111827' },
  { id: 'openai/gpt-oss-120b', label: 'GPT OSS 120B', family: 'Groq', logoText: 'O1', accent: '#a16207' },
] as const satisfies readonly ModelOption[];

export function getModelOption(modelId?: string | null): ModelOption | null {
  if (!modelId) return null;
  return TARGET_MODEL_OPTIONS.find((option) => option.id === modelId) ?? null;
}

const WEAKNESS_BADGE: Record<string, string> = {
  logical_inconsistency: 'bg-violet-100 text-violet-800 border-violet-200',
  factual_error: 'bg-sky-100 text-sky-800 border-sky-200',
  bias: 'bg-pink-100 text-pink-800 border-pink-200',
  safety_violation: 'bg-red-100 text-red-800 border-red-200',
  prompt_injection: 'bg-orange-100 text-orange-800 border-orange-200',
  hallucination: 'bg-amber-100 text-amber-800 border-amber-200',
  context_loss: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  reasoning_failure: 'bg-indigo-100 text-indigo-800 border-indigo-200',
};

export function normalizeWeaknessKey(raw: string): string {
  return raw.toLowerCase();
}

export function formatCategoryLabel(raw: string): string {
  return raw
    .toLowerCase()
    .split('_')
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export function weaknessBadgeClass(raw: string): string {
  return WEAKNESS_BADGE[normalizeWeaknessKey(raw)] ?? 'bg-slate-100 text-slate-700 border-slate-200';
}
