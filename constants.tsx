
import { Patient, BedDemandData, ResourceNeed } from './types';

export const MOCK_PATIENTS: Patient[] = [
  {
    id: "PAT-IND-001",
    name: "Aarav Sharma",
    age: 38,
    gender: "M",
    admissionTime: "2023-10-24T08:30:00",
    vitals: { heartRate: 115, systolicBP: 92, temperature: 39.1, respiratoryRate: 26, oxygenSaturation: 89 },
    labs: { wbc: 19.5, lactate: 4.2, creatinine: 1.8, crp: 145, platelets: 110, bilirubin: 1.5 },
    clinicalScores: { sofa: 8, gcs: 12, apacheII: 22 },
    allergies: ["Penicillin", "Sulfa Drugs"],
    preExistingConditions: ["Type 2 Diabetes", "Hypertension", "Chronic Smoker"],
    medications: [
      { name: "Noradrenaline", dosage: "0.1 mcg/kg/min", frequency: "Continuous Infusion", route: "IV Central", status: "Active", durationDays: 5 },
      { name: "Meropenem", dosage: "1g", frequency: "q8h", route: "IV", status: "Active", durationDays: 10 },
      { name: "Insulin (Actrapid)", dosage: "Variable", frequency: "Sliding Scale", route: "IV", status: "Active", durationDays: 14 },
      { name: "Enoxaparin", dosage: "40mg", frequency: "Once Daily", route: "SubQ", status: "Active", durationDays: 7 }
    ],
    sepsisRisk: 0.92,
    predictedLOS: 14,
    diagnosis: "Septic Shock / Multi-organ Dysfunction",
    acuityLevel: "Critical"
  },
  {
    id: "PAT-IND-002",
    name: "Ishani Gupta",
    age: 44,
    gender: "F",
    admissionTime: "2023-10-25T14:15:00",
    vitals: { heartRate: 82, systolicBP: 128, temperature: 37.0, respiratoryRate: 16, oxygenSaturation: 97 },
    labs: { wbc: 8.8, lactate: 0.9, creatinine: 0.8, crp: 8, platelets: 240, bilirubin: 0.6 },
    clinicalScores: { sofa: 1, gcs: 15, apacheII: 9 },
    allergies: ["None Known"],
    preExistingConditions: ["Coronary Artery Disease"],
    medications: [
      { name: "Aspirin", dosage: "75mg", frequency: "Once Daily", route: "Oral", status: "Active", durationDays: 30 },
      { name: "Metoprolol", dosage: "25mg", frequency: "BD", route: "Oral", status: "Active", durationDays: 30 },
      { name: "Pantoprazole", dosage: "40mg", frequency: "Once Daily", route: "Oral", status: "Active", durationDays: 14 }
    ],
    sepsisRisk: 0.08,
    predictedLOS: 3,
    diagnosis: "Post-CABG Recovery",
    acuityLevel: "Low"
  },
  {
    id: "PAT-IND-003",
    name: "Vikram Malhotra",
    age: 47,
    gender: "M",
    admissionTime: "2023-10-23T20:45:00",
    vitals: { heartRate: 98, systolicBP: 108, temperature: 37.8, respiratoryRate: 22, oxygenSaturation: 93 },
    labs: { wbc: 15.2, lactate: 2.8, creatinine: 2.4, crp: 62, platelets: 160, bilirubin: 0.9 },
    clinicalScores: { sofa: 5, gcs: 14, apacheII: 16 },
    allergies: ["Latex"],
    preExistingConditions: ["Obesity", "Mild Asthma"],
    medications: [
      { name: "Piperacillin-Tazobactam", dosage: "4.5g", frequency: "q6h", route: "IV", status: "Active", durationDays: 7 },
      { name: "Furosemide", dosage: "40mg", frequency: "BD", route: "IV", status: "Active", durationDays: 5 },
      { name: "Salbutamol Nebulizer", dosage: "2.5mg", frequency: "PRN", route: "Inhaled", status: "Active", durationDays: 3 }
    ],
    sepsisRisk: 0.61,
    predictedLOS: 8,
    diagnosis: "Acute Kidney Injury Secondary to Infection",
    acuityLevel: "Moderate"
  },
  {
    id: "PAT-IND-004",
    name: "Ananya Iyer",
    age: 42,
    gender: "F",
    admissionTime: "2023-10-26T02:00:00",
    vitals: { heartRate: 108, systolicBP: 100, temperature: 38.2, respiratoryRate: 24, oxygenSaturation: 90 },
    labs: { wbc: 17.1, lactate: 3.1, creatinine: 1.1, crp: 95, platelets: 130, bilirubin: 1.1 },
    clinicalScores: { sofa: 7, gcs: 13, apacheII: 19 },
    allergies: ["Peanuts", "Shellfish"],
    preExistingConditions: ["Hypothyroidism", "Dementia (Early Stage)"],
    medications: [
      { name: "Levothyroxine", dosage: "100mcg", frequency: "Morning", route: "Oral", status: "Active", durationDays: 90 },
      { name: "Ceftriaxone", dosage: "2g", frequency: "Once Daily", route: "IV", status: "Active", durationDays: 7 },
      { name: "Dexamethasone", dosage: "6mg", frequency: "Once Daily", route: "IV", status: "Active", durationDays: 10 }
    ],
    sepsisRisk: 0.78,
    predictedLOS: 11,
    diagnosis: "Complicated Pyelonephritis / Early ARDS",
    acuityLevel: "High"
  }
];

export const MOCK_FORECAST: BedDemandData[] = [
  { date: '24/10', currentOccupancy: 18, predictedDemand: 18, capacity: 25 },
  { date: '25/10', currentOccupancy: 20, predictedDemand: 22, capacity: 25 },
  { date: '26/10', currentOccupancy: 19, predictedDemand: 24, capacity: 25 },
  { date: '27/10', currentOccupancy: 0, predictedDemand: 26, capacity: 25 },
  { date: '28/10', currentOccupancy: 0, predictedDemand: 25, capacity: 25 },
  { date: '29/10', currentOccupancy: 0, predictedDemand: 23, capacity: 25 },
  { date: '30/10', currentOccupancy: 0, predictedDemand: 20, capacity: 25 },
];

export const MOCK_RESOURCES: ResourceNeed[] = [
  { id: 'R-01', type: 'PB840 Ventilators', current: 12, predicted: 15, utilizationRate: 0.85, replenishmentTime: '12h', status: 'Warning' },
  { id: 'R-02', type: 'CRRT Dialysis Kits', current: 8, predicted: 8, utilizationRate: 0.92, replenishmentTime: '4h', status: 'Critical' },
  { id: 'R-03', type: 'ECMO Oxygenators', current: 2, predicted: 2, utilizationRate: 0.45, replenishmentTime: '24h', status: 'Safe' },
  { id: 'R-04', type: 'Specialized Nursing (NSU)', current: 8, predicted: 11, utilizationRate: 0.98, replenishmentTime: 'N/A', status: 'Critical' },
  { id: 'R-05', type: 'IV Infusion Pumps', current: 45, predicted: 52, utilizationRate: 0.78, replenishmentTime: '6h', status: 'Warning' },
];
