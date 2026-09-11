import { useState } from "react";
import type { PantryItem } from "../types";
import { ReceiptUpload } from "../components/ReceiptUpload";
import { PantryCard } from "../components/PantryCard";

type FridgeProps = { items: PantryItem[]; onItemsAdded: (items: PantryItem[]) => void; onUpdateItem: (id: string, updates: Partial<PantryItem>) => void; onRemoveItem: (id: string) => void; onAddItem: (item: PantryItem) => void; onGenerateRecipes: () => Promise<void> | void; generatingRecipes?: boolean };

export function Fridge({ items, onUpdateItem, onRemoveItem, onAddItem, onGenerateRecipes, generatingRecipes, onItemsAdded }: FridgeProps) {
  const [draftName, setDraftName] = useState("");
  const [draftQuantity, setDraftQuantity] = useState("");
  const [draftExpiry, setDraftExpiry] = useState("");
  const [removed, setRemoved] = useState<PantryItem | null>(null);
  const [message, setMessage] = useState("");
  const [query, setQuery] = useState("");
  const visible = items.filter(item => item.name.toLowerCase().includes(query.toLowerCase()));
  return (
    <div className="pantry-page space-y-7">
      <div className="pantry-heading"><div><p className="eyebrow">YOUR EVERYDAY INGREDIENTS</p><h1>A little pantry.<br /><em>A lot of possibility.</em></h1><p>Review what you have. Make something you love.</p></div><div className="pantry-summary"><strong>{items.length.toString().padStart(2, "0")}</strong><span>ingredients in your kitchen</span><button type="button" onClick={() => void onGenerateRecipes()} disabled={generatingRecipes || !items.length} className="primary-action">{generatingRecipes ? "Finding inspiration…" : "Find recipes →"}</button></div></div>
      <div className="pantry-tools"><ReceiptUpload onItemsAdded={onItemsAdded} /><section className="manual-entry"><p className="eyebrow">JUST ONE MORE THING</p><h2>Add it yourself.</h2><p>Something missing? Make a little room for it here.</p>
        <form onSubmit={event => {
          event.preventDefault(); if (!draftName.trim()) return;
          const selectedExpiry = String(new FormData(event.currentTarget).get("expiry") ?? "");
          onAddItem({ id: `manual-${crypto.randomUUID()}`, name: draftName.trim(), quantity: draftQuantity.trim() || undefined, detectedExpiry: selectedExpiry || null, inferredExpiry: null, confidence: 1, detectionSource: "manual", expirySource: selectedExpiry ? "manual" : "none", reviewed: true, notes: "Added manually." });
          setMessage(`${draftName.trim()} added to your pantry.`); setDraftName(""); setDraftQuantity(""); setDraftExpiry("");
        }}><label>Ingredient name<input required maxLength={80} placeholder="e.g. cherry tomatoes" value={draftName} onChange={event => setDraftName(event.target.value)} /></label><div className="manual-fields"><label>Quantity<input maxLength={60} placeholder="e.g. 1 box" value={draftQuantity} onChange={event => setDraftQuantity(event.target.value)} /></label><label>Expiry date<input type="date" name="expiry" value={draftExpiry} onChange={event => setDraftExpiry(event.target.value)} /></label></div><button type="submit" className="secondary-action">+ Add ingredient</button></form>
      </section></div>
      <div className="inventory-toolbar"><h2>On the shelf <span>{items.length}</span></h2><label><span className="sr-only">Search ingredients</span><input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Find an ingredient…" /></label></div>
      {removed && <div className="undo-banner" role="status"><span>{removed.name} removed.</span><button type="button" onClick={() => { if (!items.some(item => item.id === removed.id)) onAddItem(removed); setMessage(`${removed.name} restored.`); setRemoved(null); }}>Undo removal</button></div>}
      <p className="sr-only" role="status">{message}</p>
      {!visible.length ? <div className="pantry-empty"><span aria-hidden="true">✳</span><h2>{items.length ? "Nothing on this shelf yet." : "Your next good meal starts here."}</h2><p>{items.length ? "Try a different ingredient name." : "Scan a receipt or add your first ingredient above."}</p></div> : <div className="pantry-grid">{visible.map(item => <PantryCard key={item.id} item={item} onSave={updates => { onUpdateItem(item.id, updates); setMessage(`${updates.name ?? item.name} saved.`); }} onRemove={() => { onRemoveItem(item.id); setRemoved(item); setMessage(`${item.name} removed.`); }} />)}</div>}
      <p className="pantry-storage-note">Your edits are saved on this device. Estimated dates are a guide; always check the packaging.</p>
    </div>
  );
}
