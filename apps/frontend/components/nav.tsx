"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/chat", label: "Chat" },
  { href: "/arena", label: "Arena" },
  { href: "/eval", label: "Eval" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="border-b bg-white">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center gap-6">
        <Link href="/chat" className="font-semibold tracking-tight">
          DeepEval Lab
          <span className="ml-2 text-xs font-normal text-slate-500">
            / Spanish real estate
          </span>
        </Link>
        <div className="flex gap-1 text-sm">
          {TABS.map((t) => {
            const active = path === t.href || path?.startsWith(t.href + "/");
            return (
              <Link
                key={t.href}
                href={t.href}
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  active
                    ? "bg-slate-900 text-white"
                    : "text-slate-700 hover:bg-slate-100"
                }`}
              >
                {t.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
