import { Patient, BedDemandData, ResourceNeed, XAIFactor } from "../types";

// In dev, Vite proxies /api → http://127.0.0.1:8765 (see vite.config.ts).
// Override at build/runtime by setting VITE_ICU_API_BASE.
const BASE: string = (import.meta as any).env?.VITE_ICU_API_BASE ?? "/api";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status} ${path}: ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<{ status: string; time: string }> {
  return getJSON("/health");
}

export async function fetchPatients(limit = 12): Promise<Patient[]> {
  const raw = await getJSON<any[]>(`/patients?limit=${limit}`);
  return raw.map(normalizePatient);
}

export async function fetchPatient(stayId: string): Promise<Patient> {
  const raw = await getJSON<any>(`/patients/${stayId}`);
  return normalizePatient(raw);
}

export interface ClinicalSummary {
  payload: any;
  summary_text: string;
  highlights: string[];
  fullSummary: string;
}

export async function fetchSummary(stayId: string): Promise<ClinicalSummary> {
  return getJSON<ClinicalSummary>(`/summary/${stayId}`);
}

export async function fetchBedForecast(unit?: string, days = 7): Promise<BedDemandData[]> {
  const q = new URLSearchParams();
  if (unit) q.set("unit", unit);
  q.set("days", String(days));
  return getJSON<BedDemandData[]>(`/bed_forecast?${q.toString()}`);
}

export async function fetchResources(): Promise<ResourceNeed[]> {
  return getJSON<ResourceNeed[]>("/resources");
}

export async function fetchXAI(stayId: string): Promise<XAIFactor[]> {
  return getJSON<XAIFactor[]>(`/xai/${stayId}`);
}

// Backwards-compatible API used by App.tsx and PatientAnalysis.
// runMLInference originally took a CSV row and returned a Patient via Groq;
// we now ignore the CSV row and pull a fresh batch of real patients from the
// CDSS backend so the UI flow ("Upload Records" -> analyze) still works.
export async function runMLInference(_rawRow: unknown): Promise<Patient> {
  const list = await fetchPatients(1);
  if (!list.length) throw new Error("ICU API returned no patients");
  return list[0];
}

export async function getClinicalSummary(patient: Patient): Promise<{ highlights: string[]; fullSummary: string }> {
  const s = await fetchSummary(patient.id);
  return { highlights: s.highlights, fullSummary: s.fullSummary };
}

export async function getXAIExplanation(patient: Patient): Promise<XAIFactor[]> {
  return fetchXAI(patient.id);
}

function normalizePatient(p: any): Patient {
  return {
    id: String(p.id),
    name: p.name ?? `Patient ${p.id}`,
    age: Number(p.age ?? 60),
    gender: String(p.gender ?? "M"),
    admissionTime: String(p.admissionTime ?? new Date().toISOString()),
    vitals: {
      heartRate: Number(p.vitals?.heartRate ?? 80),
      systolicBP: Number(p.vitals?.systolicBP ?? 120),
      temperature: Number(p.vitals?.temperature ?? 37),
      respiratoryRate: Number(p.vitals?.respiratoryRate ?? 16),
      oxygenSaturation: Number(p.vitals?.oxygenSaturation ?? 97),
    },
    labs: {
      wbc: Number(p.labs?.wbc ?? 9),
      lactate: Number(p.labs?.lactate ?? 1.5),
      creatinine: Number(p.labs?.creatinine ?? 1),
      crp: Number(p.labs?.crp ?? 5),
      platelets: Number(p.labs?.platelets ?? 220),
      bilirubin: Number(p.labs?.bilirubin ?? 0.7),
    },
    clinicalScores: {
      sofa: Number(p.clinicalScores?.sofa ?? 0),
      gcs: Number(p.clinicalScores?.gcs ?? 15),
      apacheII: Number(p.clinicalScores?.apacheII ?? 8),
    },
    allergies: Array.isArray(p.allergies) ? p.allergies : ["None Documented"],
    preExistingConditions: Array.isArray(p.preExistingConditions) ? p.preExistingConditions : [],
    medications: Array.isArray(p.medications) ? p.medications : [],
    sepsisRisk: Number(p.sepsisRisk ?? 0),
    predictedLOS: Number(p.predictedLOS ?? 0),
    diagnosis: String(p.diagnosis ?? "ICU Observation"),
    acuityLevel: (p.acuityLevel ?? "Low") as Patient["acuityLevel"],
  };
}
