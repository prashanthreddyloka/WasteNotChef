import { useState } from "react";
import { CheckIcon, PencilSquareIcon, TrashIcon } from "@heroicons/react/24/outline";
import type { PantryItem } from "../types";

export function PantryCard({ item, onSave, onRemove }: { item: PantryItem; onSave: (updates: Partial<PantryItem>) => void; onRemove: () => void }) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(item.name);
  const [quantity, setQuantity] = useState(item.quantity ?? "");
  const [expiry, setExpiry] = useState(item.detectedExpiry ?? item.inferredExpiry ?? "");
  const [error, setError] = useState("");
  function startEdit() { setName(item.name); setQuantity(item.quantity ?? ""); setExpiry(item.detectedExpiry ?? item.inferredExpiry ?? ""); setError(""); setEditing(true); }
  const isReceipt = item.id.startsWith("receipt-");
  const shownExpiry = item.detectedExpiry ?? item.inferredExpiry;
  return (
    <article className={`pantry-card ${editing ? "is-editing" : ""}`}>
      <div className="pantry-card-top"><span className="ingredient-monogram" aria-hidden="true">{item.name.slice(0, 1).toUpperCase()}</span><span className="pantry-source">{item.reviewed ? "Reviewed" : isReceipt ? "Receipt scan" : item.detectionSource === "manual" ? "Added by you" : "Photo scan"}</span></div>
      {editing ? <form className="pantry-edit" onSubmit={event => {
        event.preventDefault();
        if (!name.trim()) { setError("Enter an ingredient name."); return; }
        const selectedExpiry = String(new FormData(event.currentTarget).get("expiry") ?? "");
        const changedDate = selectedExpiry !== (item.detectedExpiry ?? item.inferredExpiry ?? "");
        onSave({ name: name.trim(), quantity: quantity.trim() || undefined, reviewed: true, ...(changedDate ? { detectedExpiry: selectedExpiry || null, inferredExpiry: null, expirySource: selectedExpiry ? "manual" : "none" } : {}) });
        setEditing(false);
      }}>
        <label>Ingredient name<input autoFocus required maxLength={80} value={name} onChange={event => setName(event.target.value)} /></label>
        <label>Quantity<input maxLength={60} placeholder="e.g. 2 cartons" value={quantity} onChange={event => setQuantity(event.target.value)} /></label>
        <label>Expiry date<input type="date" name="expiry" value={expiry} onChange={event => setExpiry(event.target.value)} /></label>
        {error && <p role="alert">{error}</p>}
        <div className="card-actions"><button type="submit" className="primary-action">Save changes</button><button type="button" className="text-action" onClick={() => setEditing(false)}>Cancel</button></div>
      </form> : <>
        <h2>{item.name}</h2><p className="pantry-quantity">{item.quantity || "Quantity not specified"}</p>
        <div className="expiry-note"><span>{item.inferredExpiry && !item.detectedExpiry ? "Estimated expiry" : "Expiry date"}</span><strong>{shownExpiry ? new Date(`${shownExpiry}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) : "Check the packaging"}</strong></div>
        <div className="card-actions"><button type="button" className="edit-action" onClick={startEdit} aria-label={`Edit ${item.name}`}><PencilSquareIcon className="h-4 w-4" />Edit item</button><button type="button" className="remove-action" onClick={onRemove} aria-label={`Remove ${item.name}`}><TrashIcon className="h-4 w-4" />Remove</button></div>
        {item.reviewed ? <p className="review-state"><CheckIcon className="h-3 w-3" /> Reviewed and saved on this device</p> : <button type="button" className="review-action" onClick={() => onSave({ reviewed: true })}><CheckIcon className="h-4 w-4" />Looks right — mark reviewed</button>}
      </>}
    </article>
  );
}
