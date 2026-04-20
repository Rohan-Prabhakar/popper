'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FileText, FlaskConical, LayoutDashboard } from 'lucide-react';
import { useCampaign } from '@/lib/campaign-context';

type AppSidebarProps = {
  variant: 'open' | 'closed';
};

export function AppSidebar({ variant }: AppSidebarProps) {
  const pathname = usePathname();
  const { campaignHistory } = useCampaign();
  const reportCount = campaignHistory.length;

  const navClass = (active: boolean) =>
    `flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
      active ? 'bg-slate-700/60 text-white' : 'text-slate-400 hover:bg-slate-700/30 hover:text-slate-200'
    }`;

  return (
    <aside
      className={`${
        variant === 'open' ? 'w-64' : 'w-0'
      } shrink-0 overflow-hidden border-r border-slate-700/80 bg-[var(--admin-sidebar)] text-slate-100 transition-[width] duration-200 ease-out`}
    >
      <div className="flex h-full w-64 flex-col">
        <div className="flex items-center gap-2 border-b border-slate-600/50 px-4 py-4">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--admin-primary)] text-white shadow-sm">
            <FlaskConical className="h-5 w-5" strokeWidth={2} />
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold tracking-tight">Popper RL</div>
            <div className="truncate text-xs text-slate-400">Validation console</div>
          </div>
        </div>

        <nav className="space-y-0.5 px-2 py-3">
          <Link href="/" className={navClass(pathname === '/')}>
            <LayoutDashboard className="h-4 w-4 shrink-0 text-slate-300" />
            Dashboard
          </Link>
          <Link href="/reports" className={navClass(pathname === '/reports' || pathname.startsWith('/reports/'))}>
            <FileText className="h-4 w-4 shrink-0" />
            Run reports
            {reportCount > 0 && (
              <span className="ml-auto rounded-full bg-[var(--admin-primary)] px-2 py-0.5 text-[10px] font-bold text-white">
                {reportCount}
              </span>
            )}
          </Link>
        </nav>

        <div className="mt-auto border-t border-slate-600/50 px-4 py-3 text-[10px] leading-relaxed text-slate-500">
          Extension of{' '}
          <a
            href="https://github.com/Humanitariansai/Popper"
            className="text-[var(--admin-primary)] hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            Humanitariansai/Popper
          </a>
          . UI inspired by{' '}
          <a
            href="https://github.com/PlainAdmin/plain-free-bootstrap-admin-template"
            className="text-[var(--admin-primary)] hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            PlainAdmin
          </a>
          .
        </div>
      </div>
    </aside>
  );
}
