'use client'

import { useState, useRef, useCallback } from 'react'
import type { DragEvent, ChangeEvent, FormEvent } from 'react'
import { cn } from '@/lib/utils'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Eye,
  Upload,
  X,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  Clock,
  Activity,
  ImagePlus,
  BarChart2,
  Brain,
  Stethoscope,
  Target,
  RefreshCcw,
  ShieldCheck,
} from 'lucide-react'

// ─── Types ───────────────────────────────────────────────────────────────────

type Diagnosis = 'Myopia' | 'No Myopia'

interface PredictionResult {
  diagnosis: Diagnosis
  confidence_score: number
  p_myopia: number
  inference_time_ms: number
}

interface FormValues {
  age: string
  gender: string
  baseline_se: string
  axial_length: string
}

interface FormErrors {
  age?: string
  gender?: string
  baseline_se?: string
  axial_length?: string
  image?: string
}

// ─── Constants ───────────────────────────────────────────────────────────────

const API_ENDPOINT =
  (typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL) ||
  'http://localhost:8000/predict/'

const ACCEPTED_MIME = new Set(['image/jpeg', 'image/jpg', 'image/png'])
const MAX_FILE_BYTES = 20 * 1024 * 1024 // 20 MB

const DIAGNOSIS_CONFIG: Record<
  Diagnosis,
  {
    bg: string
    border: string
    headerBg: string
    text: string
    badgeBg: string
    badgeText: string
    badgeBorder: string
    barColor: string
    Icon: React.ComponentType<{ className?: string }>
    label: string
    urgency: string
    recommendations: string[]
  }
> = {
  'Myopia': {
    bg: 'bg-rose-50/60',
    border: 'border-rose-200',
    headerBg: 'bg-rose-50',
    text: 'text-rose-700',
    badgeBg: 'bg-rose-100',
    badgeText: 'text-rose-800',
    badgeBorder: 'border-rose-200',
    barColor: 'bg-rose-500',
    Icon: AlertOctagon,
    label: 'Myopia Detected',
    urgency: 'Ophthalmology referral recommended',
    recommendations: [
      'Initiate comprehensive ophthalmologic evaluation within 4 weeks.',
      'Prescribe corrective lenses appropriate to measured refractive error.',
      'Consider myopia control: orthokeratology, low-dose atropine (0.01–0.05%), or MiSight lenses.',
      'Counsel on outdoor activity — at least 90 min/day of natural light exposure.',
      'Schedule biometry follow-up every 6 months to monitor axial length progression.',
    ],
  },
  'No Myopia': {
    bg: 'bg-emerald-50/60',
    border: 'border-emerald-200',
    headerBg: 'bg-emerald-50',
    text: 'text-emerald-700',
    badgeBg: 'bg-emerald-100',
    badgeText: 'text-emerald-800',
    badgeBorder: 'border-emerald-200',
    barColor: 'bg-emerald-500',
    Icon: CheckCircle2,
    label: 'No Myopia Detected',
    urgency: 'Routine annual follow-up',
    recommendations: [
      'No myopic changes detected. Continue routine annual screening schedule.',
      'Re-examine earlier if high-risk profile: both parents myopic, urban environment, heavy near-work.',
      'Encourage outdoor activity and near-work hygiene (20-20-20 rule, adequate lighting).',
      'Advise guardians to monitor for onset symptoms: squinting, headaches, difficulty at the board.',
    ],
  },
}

// ─── Form validation ─────────────────────────────────────────────────────────

function validate(form: FormValues, image: File | null): FormErrors {
  const errors: FormErrors = {}

  const age = Number(form.age)
  if (!form.age.trim()) {
    errors.age = 'Age is required.'
  } else if (!Number.isFinite(age) || age < 1 || age > 110) {
    errors.age = 'Age must be between 1 and 110 years.'
  }

  if (!form.gender) {
    errors.gender = 'Gender is required.'
  }

  if (form.baseline_se.trim()) {
    const se = Number(form.baseline_se)
    if (!Number.isFinite(se) || se < -30 || se > 10) {
      errors.baseline_se = 'Value must be between −30.00 and +10.00 D.'
    }
  }

  if (form.axial_length.trim()) {
    const al = Number(form.axial_length)
    if (!Number.isFinite(al) || al < 18 || al > 40) {
      errors.axial_length = 'Value must be between 18.00 and 40.00 mm.'
    }
  }

  if (!image) {
    errors.image = 'A fundus photograph is required.'
  }

  return errors
}

// ─────────────────────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────────────────────

export default function MyopiaDashboard() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [form, setForm] = useState<FormValues>({
    age: '',
    gender: '',
    baseline_se: '',
    axial_length: '',
  })
  const [image, setImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [fieldErrors, setFieldErrors] = useState<FormErrors>({})

  const fileInputRef = useRef<HTMLInputElement>(null)

  // ── File handling ──────────────────────────────────────────────────────────

  const acceptFile = useCallback((file: File) => {
    if (!ACCEPTED_MIME.has(file.type)) {
      setFieldErrors(prev => ({
        ...prev,
        image: 'Only JPEG and PNG images are accepted.',
      }))
      return
    }
    if (file.size > MAX_FILE_BYTES) {
      setFieldErrors(prev => ({
        ...prev,
        image: 'Image file must be smaller than 20 MB.',
      }))
      return
    }
    setImage(file)
    setFieldErrors(prev => ({ ...prev, image: undefined }))
    const reader = new FileReader()
    reader.onload = e => setImagePreview(e.target?.result as string)
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) acceptFile(file)
    },
    [acceptFile],
  )

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleFileInput = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) acceptFile(file)
    },
    [acceptFile],
  )

  const clearImage = useCallback(() => {
    setImage(null)
    setImagePreview(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  // ── Form field helpers ────────────────────────────────────────────────────

  const handleTextField =
    (field: keyof Omit<FormValues, 'gender'>) =>
    (e: ChangeEvent<HTMLInputElement>) => {
      setForm(prev => ({ ...prev, [field]: e.target.value }))
      setFieldErrors(prev => ({ ...prev, [field]: undefined }))
    }

  const handleGender = (value: string) => {
    setForm(prev => ({ ...prev, gender: value }))
    setFieldErrors(prev => ({ ...prev, gender: undefined }))
  }

  // ── Reset ─────────────────────────────────────────────────────────────────

  const handleReset = useCallback(() => {
    setForm({ age: '', gender: '', baseline_se: '', axial_length: '' })
    setImage(null)
    setImagePreview(null)
    setResult(null)
    setApiError(null)
    setFieldErrors({})
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  // ── Submit ────────────────────────────────────────────────────────────────

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault()

      const errors = validate(form, image)
      if (Object.keys(errors).length > 0) {
        setFieldErrors(errors)
        return
      }

      setIsLoading(true)
      setApiError(null)
      setResult(null)

      const payload = new FormData()
      payload.append('fundus_image', image!)
      payload.append('age', form.age)
      payload.append('gender', form.gender === 'male' ? 'M' : 'F')
      payload.append('baseline_se', form.baseline_se)
      payload.append('axial_length', form.axial_length)

      try {
        const res = await fetch(API_ENDPOINT, { method: 'POST', body: payload })

        if (!res.ok) {
          let detail = `Server error (HTTP ${res.status})`
          try {
            const body = await res.json()
            if (typeof body?.detail === 'string') detail = body.detail
          } catch {
            /* keep default */
          }
          throw new Error(detail)
        }

        const data: PredictionResult = await res.json()
        setResult(data)
      } catch (err) {
        setApiError(
          err instanceof Error
            ? err.message
            : 'Unexpected network error. Ensure the inference server is running on port 8000.',
        )
      } finally {
        setIsLoading(false)
      }
    },
    [form, image],
  )

  // ── Derived ───────────────────────────────────────────────────────────────

  const riskConfig = result ? DIAGNOSIS_CONFIG[result.diagnosis] : null
  const confidencePct = result ? Math.round(result.confidence_score * 100) : 0
  const pMyopiaPct = result ? Math.round(result.p_myopia * 100) : 0
  const showReset = result !== null || apiError !== null

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 shadow-sm backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            {/* Brand */}
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 shadow-sm">
                <Eye className="h-5 w-5 text-white" />
              </div>
              <div className="leading-tight">
                <p className="text-sm font-bold tracking-tight text-slate-900">
                  ZhanAI
                </p>
                <p className="text-[11px] text-slate-500">
                  Clinical Decision Support System
                </p>
              </div>
            </div>

            {/* Status badge */}
            <div className="flex items-center gap-2">
              <span className="hidden sm:flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                Inference Server Online
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Page heading ───────────────────────────────────────────────────── */}
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex items-start gap-3">
            <div className="rounded-md bg-blue-50 p-2 mt-0.5 shrink-0">
              <Stethoscope className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-slate-900">
                Myopia Screening Assessment
              </h1>
              <p className="mt-0.5 text-sm text-slate-500 max-w-2xl">
                Enter patient demographics and upload a retinal fundus
                photograph. The multimodal AI model (SwinV2-Tiny&nbsp;+&nbsp;MLP
                fusion, trained on ODIR-5K, val AUC&nbsp;0.946) will classify
                the eye as &ldquo;Myopia&rdquo; or &ldquo;No Myopia&rdquo;
                with a confidence score.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Main ───────────────────────────────────────────────────────────── */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 sm:px-6 lg:px-8 py-8">
        <form onSubmit={handleSubmit} noValidate>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">

            {/* ── Left column ─────────────────────────────────────────────── */}
            <div className="lg:col-span-3 space-y-6">

              {/* Patient parameters card */}
              <Card className="shadow-sm border-slate-200">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-2 text-base font-semibold text-slate-900">
                    <Activity className="h-4 w-4 text-blue-600 shrink-0" />
                    Patient Clinical Parameters
                  </CardTitle>
                  <CardDescription className="text-xs">
                    All fields are required. Values feed the tabular MLP branch
                    of the fusion model.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">

                    {/* Age */}
                    <div className="space-y-1.5">
                      <Label
                        htmlFor="age"
                        className="text-sm font-medium text-slate-700"
                      >
                        Age{' '}
                        <span className="font-normal text-slate-400">
                          (years, 1–110)
                        </span>
                      </Label>
                      <Input
                        id="age"
                        type="number"
                        min={1}
                        max={110}
                        step={1}
                        placeholder="e.g. 45"
                        value={form.age}
                        onChange={handleTextField('age')}
                        className={cn(
                          'h-9 text-sm',
                          fieldErrors.age &&
                            'border-red-400 focus-visible:ring-red-400',
                        )}
                      />
                      {fieldErrors.age ? (
                        <p className="text-xs text-red-600">
                          {fieldErrors.age}
                        </p>
                      ) : (
                        <p className="text-xs text-slate-400">
                          ODIR-5K cohort range: 1–110 years
                        </p>
                      )}
                    </div>

                    {/* Gender */}
                    <div className="space-y-1.5">
                      <Label className="text-sm font-medium text-slate-700">
                        Biological Sex
                      </Label>
                      <Select
                        value={form.gender}
                        onValueChange={handleGender}
                      >
                        <SelectTrigger
                          className={cn(
                            'h-9 text-sm',
                            fieldErrors.gender &&
                              'border-red-400 focus:ring-red-400',
                          )}
                        >
                          <SelectValue placeholder="Select…" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="male">Male</SelectItem>
                          <SelectItem value="female">Female</SelectItem>
                        </SelectContent>
                      </Select>
                      {fieldErrors.gender && (
                        <p className="text-xs text-red-600">
                          {fieldErrors.gender}
                        </p>
                      )}
                    </div>

                    {/* Baseline SE */}
                    <div className="space-y-1.5">
                      <Label
                        htmlFor="baseline_se"
                        className="text-sm font-medium text-slate-700"
                      >
                        Baseline Spherical Equivalent{' '}
                        <span className="font-normal text-slate-400">(D, optional)</span>
                      </Label>
                      <Input
                        id="baseline_se"
                        type="number"
                        step={0.25}
                        min={-30}
                        max={10}
                        placeholder="e.g. −3.50"
                        value={form.baseline_se}
                        onChange={handleTextField('baseline_se')}
                        className={cn(
                          'h-9 text-sm',
                          fieldErrors.baseline_se &&
                            'border-red-400 focus-visible:ring-red-400',
                        )}
                      />
                      {fieldErrors.baseline_se ? (
                        <p className="text-xs text-red-600">
                          {fieldErrors.baseline_se}
                        </p>
                      ) : (
                        <p className="text-xs text-slate-400">
                          Negative = myopic &nbsp;·&nbsp; step 0.25 D
                        </p>
                      )}
                    </div>

                    {/* Axial Length */}
                    <div className="space-y-1.5">
                      <Label
                        htmlFor="axial_length"
                        className="text-sm font-medium text-slate-700"
                      >
                        Axial Length{' '}
                        <span className="font-normal text-slate-400">(mm, optional)</span>
                      </Label>
                      <Input
                        id="axial_length"
                        type="number"
                        step={0.01}
                        min={18}
                        max={35}
                        placeholder="e.g. 25.43"
                        value={form.axial_length}
                        onChange={handleTextField('axial_length')}
                        className={cn(
                          'h-9 text-sm',
                          fieldErrors.axial_length &&
                            'border-red-400 focus-visible:ring-red-400',
                        )}
                      />
                      {fieldErrors.axial_length ? (
                        <p className="text-xs text-red-600">
                          {fieldErrors.axial_length}
                        </p>
                      ) : (
                        <p className="text-xs text-slate-400">
                          Normal range: 22–27 mm &nbsp;·&nbsp; step 0.01 mm
                        </p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Fundus image upload card */}
              <Card className="shadow-sm border-slate-200">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-2 text-base font-semibold text-slate-900">
                    <ImagePlus className="h-4 w-4 text-blue-600 shrink-0" />
                    Fundus Photograph
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Upload a retinal fundus image (JPEG or PNG, max&nbsp;20&nbsp;MB).
                    The SwinV2-Tiny vision branch expects a 256&nbsp;×&nbsp;256
                    input — resizing is handled automatically.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {imagePreview ? (
                    /* Preview state */
                    <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-950">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={imagePreview}
                        alt="Fundus photograph preview"
                        className="h-52 w-full object-contain"
                      />
                      <div className="flex items-center justify-between border-t border-slate-800 bg-slate-900 px-4 py-2">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="h-2 w-2 shrink-0 rounded-full bg-emerald-400" />
                          <span className="truncate text-xs font-medium text-slate-300">
                            {image?.name}
                          </span>
                          <span className="shrink-0 text-xs text-slate-500">
                            ({image ? (image.size / 1024).toFixed(0) : 0} KB)
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={clearImage}
                          aria-label="Remove uploaded image"
                          className="ml-3 shrink-0 rounded p-1 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* Drag-and-drop zone */
                    <div
                      role="button"
                      tabIndex={0}
                      aria-label="Click or drag to upload a fundus photograph"
                      onClick={() => fileInputRef.current?.click()}
                      onKeyDown={e =>
                        e.key === 'Enter' && fileInputRef.current?.click()
                      }
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      className={cn(
                        'flex h-52 cursor-pointer select-none flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2',
                        isDragging
                          ? 'scale-[1.01] border-blue-400 bg-blue-50'
                          : fieldErrors.image
                            ? 'border-red-300 bg-red-50/40 hover:border-red-400'
                            : 'border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/40',
                      )}
                    >
                      <div
                        className={cn(
                          'rounded-full p-3.5 transition-colors',
                          isDragging ? 'bg-blue-100' : 'bg-slate-100',
                        )}
                      >
                        <Upload
                          className={cn(
                            'h-6 w-6 transition-colors',
                            isDragging ? 'text-blue-600' : 'text-slate-400',
                          )}
                        />
                      </div>
                      <div className="text-center">
                        <p className="text-sm font-medium text-slate-700">
                          {isDragging
                            ? 'Release to upload'
                            : 'Drop fundus image here'}
                        </p>
                        <p className="mt-0.5 text-xs text-slate-400">
                          or{' '}
                          <span className="text-blue-600 underline underline-offset-2">
                            browse files
                          </span>
                          &nbsp;·&nbsp;JPEG, PNG&nbsp;·&nbsp;max 20 MB
                        </p>
                      </div>
                    </div>
                  )}

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/jpg,image/png"
                    className="sr-only"
                    tabIndex={-1}
                    onChange={handleFileInput}
                  />

                  {fieldErrors.image && (
                    <p className="mt-2 text-xs text-red-600">
                      {fieldErrors.image}
                    </p>
                  )}
                </CardContent>
              </Card>

              {/* Action row */}
              <div className="flex items-center gap-3">
                <Button
                  type="submit"
                  disabled={isLoading}
                  className="h-10 flex-1 gap-2 bg-blue-600 font-medium text-white hover:bg-blue-700 focus-visible:ring-blue-600"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analysing…
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4" />
                      Run Progression Analysis
                    </>
                  )}
                </Button>

                {showReset && (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleReset}
                    className="h-10 gap-2 border-slate-300 text-slate-600 hover:bg-slate-50"
                  >
                    <RefreshCcw className="h-4 w-4" />
                    Reset
                  </Button>
                )}
              </div>
            </div>

            {/* ── Right column ────────────────────────────────────────────── */}
            <div className="lg:col-span-2 space-y-5">

              {/* API error */}
              {apiError && (
                <Card className="border-red-200 bg-red-50 shadow-sm">
                  <CardContent className="pt-5">
                    <div className="flex gap-3">
                      <AlertOctagon className="mt-0.5 h-5 w-5 shrink-0 text-red-500" />
                      <div>
                        <p className="text-sm font-semibold text-red-800">
                          Analysis Failed
                        </p>
                        <p className="mt-1 text-xs leading-relaxed text-red-600">
                          {apiError}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Results card */}
              {result && riskConfig ? (
                <Card
                  className={cn(
                    'overflow-hidden shadow-sm border',
                    riskConfig.border,
                  )}
                >
                  {/* Coloured header strip */}
                  <CardHeader
                    className={cn('border-b pb-4', riskConfig.headerBg, riskConfig.border)}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <riskConfig.Icon
                          className={cn('h-6 w-6 shrink-0', riskConfig.text)}
                        />
                        <div>
                          <CardTitle
                            className={cn(
                              'text-lg font-bold leading-tight',
                              riskConfig.text,
                            )}
                          >
                            {riskConfig.label}
                          </CardTitle>
                          <CardDescription className="text-xs text-slate-500">
                            {riskConfig.urgency}
                          </CardDescription>
                        </div>
                      </div>
                      <span
                        className={cn(
                          'shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-semibold',
                          riskConfig.badgeBg,
                          riskConfig.badgeText,
                          riskConfig.badgeBorder,
                        )}
                      >
                        AI Result
                      </span>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-5 pt-5">
                    {/* Confidence score */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="flex items-center gap-1.5 text-sm font-medium text-slate-600">
                          <BarChart2 className="h-3.5 w-3.5 text-slate-400" />
                          Model Confidence
                        </span>
                        <span
                          className={cn(
                            'tabular-nums text-sm font-bold',
                            riskConfig.text,
                          )}
                        >
                          {confidencePct}%
                        </span>
                      </div>
                      {/* Custom progress bar for full color control */}
                      <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          role="progressbar"
                          aria-valuenow={confidencePct}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          className={cn(
                            'h-full rounded-full transition-all duration-700 ease-out',
                            riskConfig.barColor,
                          )}
                          style={{ width: `${confidencePct}%` }}
                        />
                      </div>
                      <p className="text-[11px] text-slate-400">
                        Softmax probability of the predicted class
                      </p>
                    </div>

                    <div className="border-t border-slate-100" />

                    {/* Inference time */}
                    <div className="flex items-center justify-between rounded-md bg-slate-50 px-3.5 py-2.5">
                      <span className="flex items-center gap-2 text-xs text-slate-500">
                        <Clock className="h-3.5 w-3.5" />
                        Inference Time
                      </span>
                      <span className="tabular-nums text-xs font-semibold text-slate-700">
                        {result.inference_time_ms.toFixed(1)}&nbsp;ms
                      </span>
                    </div>

                    {/* P(myopia) stat */}
                    <div className="flex items-center justify-between rounded-md bg-slate-50 px-3.5 py-2.5">
                      <span className="flex items-center gap-2 text-xs text-slate-500">
                        <Activity className="h-3.5 w-3.5" />
                        P(Myopia)
                      </span>
                      <span className={cn('tabular-nums text-xs font-semibold', riskConfig.text)}>
                        {pMyopiaPct}%
                      </span>
                    </div>

                    {/* Clinical recommendations */}
                    <div
                      className={cn(
                        'rounded-md border p-3.5',
                        riskConfig.border,
                        riskConfig.bg,
                      )}
                    >
                      <p className="mb-2 text-xs font-semibold text-slate-600">
                        Clinical Recommendations
                      </p>
                      <ul className="space-y-1.5">
                        {riskConfig.recommendations.map((rec, i) => (
                          <li key={i} className={cn('flex items-start gap-2 text-xs leading-relaxed', riskConfig.text)}>
                            <span className="mt-0.5 shrink-0">›</span>
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Disclaimer */}
                    <div className="flex gap-2 rounded-md border border-slate-100 bg-slate-50 p-3">
                      <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-400" />
                      <p className="text-[10px] leading-relaxed text-slate-400">
                        This prediction is for{' '}
                        <strong className="font-medium text-slate-500">
                          clinical decision support only
                        </strong>{' '}
                        and must be reviewed by a qualified ophthalmologist
                        before any clinical action is taken. Not a substitute
                        for professional medical judgement.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                /* Empty state — shown before first submission */
                !apiError && (
                  <Card className="border-dashed border-slate-200 shadow-sm">
                    <CardContent className="flex flex-col items-center justify-center py-16 text-center">
                      <div className="mb-4 rounded-full bg-slate-100 p-4">
                        <Target className="h-9 w-9 text-slate-300" />
                      </div>
                      <p className="text-sm font-medium text-slate-500">
                        No analysis performed yet
                      </p>
                      <p className="mt-1.5 max-w-[200px] text-xs leading-relaxed text-slate-400">
                        Fill in the patient parameters, upload a fundus image,
                        and click &ldquo;Run Progression Analysis&rdquo;.
                      </p>
                    </CardContent>
                  </Card>
                )
              )}

              {/* Model info card */}
              <Card className="border-slate-200 bg-blue-50/40 shadow-sm">
                <CardContent className="pt-5">
                  <h3 className="mb-3 flex items-center gap-2 text-xs font-semibold text-slate-700">
                    <Brain className="h-3.5 w-3.5 text-blue-500" />
                    Model Architecture
                  </h3>
                  <ul className="space-y-2.5">
                    {(
                      [
                        ['Dataset', 'ODIR-5K (6392 fundus images)'],
                        ['Vision branch', 'SwinV2-Tiny (256×256 → 768-d)'],
                        ['Tabular branch', 'MLP: 2 → 128 → 64'],
                        ['Fusion', 'Concat 832 → Linear 256 → 2'],
                        ['Classes', 'No Myopia · Myopia'],
                        ['Val AUC', '0.9463 (epoch 4)'],
                        ['Loss', 'Focal Loss (γ=2, α=0.25)'],
                        ['Precision', 'Mixed-precision (AMP)'],
                      ] as [string, string][]
                    ).map(([label, value]) => (
                      <li key={label} className="flex justify-between gap-3">
                        <span className="text-xs text-slate-500">{label}</span>
                        <span className="text-right text-xs font-medium text-slate-700">
                          {value}
                        </span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>
          </div>
        </form>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="mt-auto border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-4">
          <p className="text-center text-[11px] text-slate-400">
            ZhanAI &nbsp;·&nbsp; AI-Assisted Clinical Decision Support
            &nbsp;·&nbsp; For investigational use only &nbsp;·&nbsp;
            <span className="text-slate-500">Not FDA / CE cleared</span>
          </p>
        </div>
      </footer>
    </div>
  )
}
