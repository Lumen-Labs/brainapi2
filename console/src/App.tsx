import { Navigate, Route, Routes } from "react-router-dom";
import { getSession } from "./lib/api";
import Layout from "./components/Layout";
import LoginPage from "./pages/LoginPage";
import OverviewPage from "./pages/OverviewPage";
import GraphPage from "./pages/GraphPage";
import DataPage from "./pages/DataPage";
import ObservationsPage from "./pages/ObservationsPage";
import VectorsPage from "./pages/VectorsPage";
import TasksPage from "./pages/TasksPage";
import IngestPage from "./pages/IngestPage";

function RequireAuth({ children }: { children: React.ReactNode }) {
  if (!getSession()) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        <Route index element={<OverviewPage />} />
        <Route path="graph" element={<GraphPage />} />
        <Route path="data" element={<DataPage />} />
        <Route path="observations" element={<ObservationsPage />} />
        <Route path="vectors" element={<VectorsPage />} />
        <Route path="tasks" element={<TasksPage />} />
        <Route path="ingest" element={<IngestPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  );
}
