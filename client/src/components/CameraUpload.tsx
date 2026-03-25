import { CameraIcon, PhotoIcon } from "@heroicons/react/24/outline";
import { motion } from "framer-motion";
import { useRef, useState } from "react";

type CameraUploadProps = {
  onFileSelected: (file: File) => Promise<void> | void;
  busy?: boolean;
};

export function CameraUpload({ onFileSelected, busy }: CameraUploadProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const cameraRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  async function handleFiles(fileList: FileList | null) {
    const file = fileList?.[0];
    if (file) {
      await onFileSelected(file);
    }
  }

  return (
    <motion.div
      whileHover={{ y: -4 }}
      className={`rounded-[2rem] border border-white/60 bg-white/75 p-5 shadow-float backdrop-blur ${dragging ? "ring-2 ring-teal-500" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        void handleFiles(event.dataTransfer.files);
      }}
    >
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-teal-700">Snap to plan</p>
          <h3 className="mt-2 font-display text-2xl text-ink">Upload a fridge photo</h3>
        </div>
        <span className="rounded-full bg-glow px-3 py-1 text-xs font-semibold text-teal-900">Camera-first</span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <button
          type="button"
          disabled={busy}
          onClick={() => cameraRef.current?.click()}
          className="flex min-h-40 flex-col items-center justify-center rounded-[1.75rem] bg-ink text-white transition hover:bg-slate-900"
        >
          <CameraIcon className="mb-3 h-8 w-8" />
          <span className="text-lg font-semibold">Open camera</span>
          <span className="mt-1 text-sm text-white/70">Use direct capture on supported phones</span>
        </button>

        <button
          type="button"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
          className="flex min-h-40 flex-col items-center justify-center rounded-[1.75rem] border border-slate-200 bg-oat transition hover:border-coral/50 hover:bg-white"
        >
          <PhotoIcon className="mb-3 h-8 w-8 text-coral" />
          <span className="text-lg font-semibold text-ink">Choose a file</span>
          <span className="mt-1 text-sm text-slate-500">Fallback picker and drag-and-drop</span>
        </button>
      </div>

      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(event) => void handleFiles(event.target.files)}
      />
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => void handleFiles(event.target.files)}
      />
    </motion.div>
  );
}
