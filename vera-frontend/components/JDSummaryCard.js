export default function JDSummaryCard({ jd }) {
    if (!jd) return null;
  
    const mandatory = jd.mandatory_skills || [];
    const preferred = [...(jd.preferred_technical_skills || []), ...(jd.soft_preferred_skills || [])];
  
    const eduReq = (jd.education_requirements || [])
      .map((e) => [e.degree_level, e.field].filter(Boolean).join(" in "))
      .filter(Boolean)
      .join(" or ");
  
    const summaryLine = [
      jd.min_years_experience != null ? `${jd.min_years_experience}+ yrs experience` : null,
      eduReq || null,
    ]
      .filter(Boolean)
      .join(" · ");
  
    return (
      <div className="panel" style={{ marginTop: 14 }}>
        <div className="body">
          <h3 style={{ margin: "0 0 12px" }}>{jd.role_title || "Role"}</h3>
  
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: mandatory.length || preferred.length ? 12 : 0 }}>
            {mandatory.map((skill) => (
              <span key={`m-${skill}`} className="tag" style={{ background: "var(--b)", color: "#fff", fontWeight: 600 }}>
                {skill}
              </span>
            ))}
            {preferred.map((skill) => (
              <span key={`p-${skill}`} className="tag pref">
                {skill}
              </span>
            ))}
          </div>
  
          {summaryLine && <div style={{ color: "var(--m)", fontSize: 13 }}>{summaryLine}</div>}
        </div>
      </div>
    );
  }
  