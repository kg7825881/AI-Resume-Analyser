"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAppState } from "../app/providers";

const ITEMS = [
  { href: "/screen", icon: "◈", label: "Screen Candidates", match: (p) => p.startsWith("/screen") },
  { href: "/results", icon: "▤", label: "Candidate Ranking", match: (p) => p.startsWith("/results") },
  { href: "/candidate", icon: "◎", label: "Candidate Detail", match: (p) => p.startsWith("/candidate") },
];

export default function Sidebar() {
  const pathname = usePathname() || "/";
  const { state } = useAppState();

  function hrefFor(item) {
    if (item.href === "/results" && state.currentRole) return `/results/${state.currentRole.role_id}`;
    return item.href;
  }

  return (
    <aside className="side">
      <div className="logo">
        <div className="mark">TL</div>
        <div>
          <b>TalentLens</b>
          <small>AI Screening Agent</small>
        </div>
      </div>
      <nav className="nav">
        {ITEMS.map((item) => (
          <Link
            key={item.href}
            href={hrefFor(item)}
            className={item.match(pathname) ? "active" : ""}
          >
            {item.icon} <span>{item.label}</span>
          </Link>
        ))}
      </nav>
      <div className="side-footer">
        {state.currentRole ? (
          <>
            Active role
            <br />
            <b style={{ color: "#d8e5f1" }}>{state.currentRole.role_title}</b>
          </>
        ) : (
          "No role selected yet"
        )}
      </div>
    </aside>
  );
}