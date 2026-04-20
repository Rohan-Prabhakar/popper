'use client';

import GroqMono from '@lobehub/icons/es/Groq/components/Mono';
import MetaMono from '@lobehub/icons/es/Meta/components/Mono';
import OpenAIMono from '@lobehub/icons/es/OpenAI/components/Mono';
import QwenColor from '@lobehub/icons/es/Qwen/components/Color';
import { getModelOption } from '@/lib/campaign';

function ModelLogo({
  modelId,
  size,
}: {
  modelId?: string | null;
  size: number;
}) {
  const normalized = (modelId || '').toLowerCase();

  if (normalized.includes('qwen')) {
    return <QwenColor size={size} />;
  }

  if (normalized.includes('gpt') || normalized.includes('openai')) {
    return <OpenAIMono size={size} />;
  }

  if (normalized.includes('llama')) {
    return <MetaMono size={size} />;
  }

  return <GroqMono size={size} />;
}

export function ModelBadge({
  modelId,
  compact = false,
}: {
  modelId?: string | null;
  compact?: boolean;
}) {
  const model = getModelOption(modelId);
  const label = model?.label ?? modelId ?? 'Unknown model';

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 ${
        compact ? 'px-2.5 py-1' : 'px-3 py-1.5'
      } shadow-sm`}
    >
      <span
        className={`inline-flex items-center justify-center rounded-full bg-slate-50 ${
          compact ? 'h-6 w-6' : 'h-8 w-8'
        }`}
      >
        <ModelLogo modelId={modelId} size={compact ? 14 : 18} />
      </span>
      <span className={`${compact ? 'text-xs' : 'text-sm'} font-medium text-slate-700`}>{label}</span>
    </div>
  );
}
