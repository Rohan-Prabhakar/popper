'use client';

import { useState } from 'react';
import { PanelLeft } from 'lucide-react';
import { AppSidebar } from '@/components/app-sidebar';

export function AppShell({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="min-h-screen flex bg-[var(--admin-surface)]">
      <AppSidebar variant={sidebarOpen ? 'open' : 'closed'} />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="sticky top-0 z-20 flex items-center gap-2 border-b border-[var(--admin-border)] bg-[var(--admin-card)] px-3 py-2 md:hidden">
          <button
            type="button"
            onClick={() => setSidebarOpen((o) => !o)}
            className="rounded-lg border border-slate-200 p-2 text-slate-600 hover:bg-slate-50"
            aria-label="Toggle sidebar"
          >
            <PanelLeft className="h-5 w-5" />
          </button>
          <span className="text-sm font-medium text-slate-700">Menu</span>
        </div>
        {children}
      </div>
    </div>
  );
}
