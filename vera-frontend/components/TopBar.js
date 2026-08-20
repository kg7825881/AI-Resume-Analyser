"use client";

const CRUMBS = {
  "/screen": "New screening",
  "/results": "Candidate ranking",
  "/candidate": "Candidate detail",
};

import { usePathname } from "next/navigation";

export default function TopBar() {
  const pathname = usePathname() || "/";
  const section = "/" + (pathname.split("/")[1] || "screen");
  const crumb = CRUMBS[section] || "New screening";

  return (
    <div className="top">
      <span>
        Recruitment / AI Screening / <b>{crumb}</b>
      </span>
      <span className="user">HR Workspace · ●</span>
    </div>
  );
}