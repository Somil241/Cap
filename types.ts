
export interface Medication {
  name: string;
  dosage: string;
  frequency: string;
  route: string;
  status: 'Active' | 'Discontinued' | 'Completed';
  durationDays: number;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  admissionTime: string;
  vitals: {
    heartRate: number;
    systolicBP: number;
    temperature: number;
    respiratoryRate: number;
    oxygenSaturation: number;
  };
  labs: {
    wbc: number;
    lactate: number;
    creatinine: number;
    crp: number;
    platelets: number;
    bilirubin: number;
  };
  clinicalScores: {
    sofa: number;
    gcs: number;
    apacheII: number;
  };
  allergies: string[];
  preExistingConditions: string[];
  medications: Medication[];
  sepsisRisk: number; 
  predictedLOS: number; 
  diagnosis: string;
  acuityLevel: 'Low' | 'Moderate' | 'High' | 'Critical';
}

export interface BedDemandData {
  date: string;
  currentOccupancy: number;
  predictedDemand: number;
  capacity: number;
}

export interface ResourceNeed {
  type: string;
  id: string;
  current: number;
  predicted: number;
  utilizationRate: number;
  replenishmentTime: string;
  status: 'Safe' | 'Warning' | 'Critical';
}

export interface XAIFactor {
  feature: string;
  contribution: number; 
  impact: 'High' | 'Medium' | 'Low';
}
