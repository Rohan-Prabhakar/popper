'use client';

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { getClientBackendUrl, type CampaignResult } from '@/lib/campaign';

const LAST_RUN_STORAGE_KEY = 'popper_rl_last_campaign_v3';
const HISTORY_STORAGE_KEY = 'popper_rl_campaign_history_v3';
const MAX_HISTORY_ITEMS = 25;

type CampaignContextValue = {
  campaignResult: CampaignResult | null;
  campaignHistory: CampaignResult[];
  setCampaignResult: React.Dispatch<React.SetStateAction<CampaignResult | null>>;
  getCampaignByRunId: (runId: string) => CampaignResult | null;
};

const CampaignContext = createContext<CampaignContextValue | null>(null);

function isCampaignResult(value: unknown): value is CampaignResult {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as CampaignResult;
  return (
    typeof candidate.total_tests === 'number' &&
    Array.isArray(candidate.validation_reports)
  );
}

function loadStoredCampaign(): CampaignResult | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(LAST_RUN_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return isCampaignResult(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function loadStoredHistory(): CampaignResult[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isCampaignResult);
  } catch {
    return [];
  }
}

function mergeCampaignIntoHistory(
  history: CampaignResult[],
  campaign: CampaignResult | null,
): CampaignResult[] {
  if (!campaign) return history;

  const runId = campaign.campaign_meta?.run_id;
  const next = history.filter((entry) => entry.campaign_meta?.run_id !== runId);
  next.unshift(campaign);
  return next.slice(0, MAX_HISTORY_ITEMS);
}

function sortHistory(history: CampaignResult[]): CampaignResult[] {
  return [...history].sort((a, b) => {
    const aTime = a.campaign_meta?.completed_at ?? '';
    const bTime = b.campaign_meta?.completed_at ?? '';
    return bTime.localeCompare(aTime);
  });
}

export function CampaignProvider({ children }: { children: React.ReactNode }) {
  const [campaignResult, setCampaignResultState] = useState<CampaignResult | null>(null);
  const [campaignHistory, setCampaignHistory] = useState<CampaignResult[]>([]);

  useEffect(() => {
    setCampaignResultState(loadStoredCampaign());
    setCampaignHistory(loadStoredHistory());
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function syncFromBackend() {
      try {
        const response = await fetch(`${getClientBackendUrl()}/api/runs`, { cache: 'no-store' });
        if (!response.ok) return;
        const payload = await response.json();
        const runs: CampaignResult[] = Array.isArray(payload?.runs)
          ? payload.runs.filter(isCampaignResult)
          : [];
        if (cancelled || runs.length === 0) return;

        const merged = sortHistory(
          runs.reduce(
            (acc, run) => mergeCampaignIntoHistory(acc, run),
            loadStoredHistory(),
          ),
        );

        setCampaignHistory(merged);
        setCampaignResultState((prev) => prev ?? merged[0] ?? null);

        try {
          localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(merged));
          if (merged[0]) localStorage.setItem(LAST_RUN_STORAGE_KEY, JSON.stringify(merged[0]));
        } catch {
          /* ignore */
        }
      } catch {
        /* ignore */
      }
    }

    syncFromBackend();
    return () => {
      cancelled = true;
    };
  }, []);

  const setCampaignResult = useCallback((update: React.SetStateAction<CampaignResult | null>) => {
    setCampaignResultState((prev) => {
      const next =
        typeof update === 'function'
          ? (update as (p: CampaignResult | null) => CampaignResult | null)(prev)
          : update;

      setCampaignHistory((historyPrev) => {
        const merged = sortHistory(mergeCampaignIntoHistory(historyPrev, next));
        if (typeof window !== 'undefined') {
          try {
            localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(merged));
          } catch {
            /* ignore */
          }
        }
        return merged;
      });

      if (typeof window !== 'undefined') {
        try {
          if (next) localStorage.setItem(LAST_RUN_STORAGE_KEY, JSON.stringify(next));
          else localStorage.removeItem(LAST_RUN_STORAGE_KEY);
        } catch {
          /* ignore */
        }
      }
      return next;
    });
  }, []);

  const getCampaignByRunId = useCallback(
    (runId: string) =>
      campaignHistory.find((entry) => entry.campaign_meta?.run_id === runId) ??
      (campaignResult?.campaign_meta?.run_id === runId ? campaignResult : null),
    [campaignHistory, campaignResult],
  );

  const value = useMemo(
    () => ({ campaignResult, campaignHistory, setCampaignResult, getCampaignByRunId }),
    [campaignHistory, campaignResult, getCampaignByRunId, setCampaignResult],
  );

  return <CampaignContext.Provider value={value}>{children}</CampaignContext.Provider>;
}

export function useCampaign() {
  const ctx = useContext(CampaignContext);
  if (!ctx) {
    throw new Error('useCampaign must be used within CampaignProvider');
  }
  return ctx;
}
