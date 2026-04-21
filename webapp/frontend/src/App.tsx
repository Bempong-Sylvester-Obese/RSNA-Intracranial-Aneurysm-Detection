import { Navigate, Route, Routes } from "react-router-dom";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";
import Upload from "@/pages/Upload";
import Viewer from "@/pages/Viewer";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <DisclaimerBanner />
      <Routes>
        <Route path="/" element={<Upload />} />
        <Route path="/viewer/:seriesId" element={<Viewer />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
