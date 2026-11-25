import { useCallback, useEffect, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  AppShell,
  AppShellMain,
  AppShellRail,
  AppShellSidebar,
  AppShellSidebarContent,
  AppShellSidebarFooter,
  AppShellSidebarHeader,
  Button,
  DescriptionDetails,
  DescriptionList,
  DescriptionTerm,
  Disclosure,
  DisclosureContent,
  DisclosureTrigger,
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerTitle,
  DrawerTrigger,
  Field,
  FieldDescription,
  FieldLabel,
  GlobalHeader,
  GlobalHeaderActions,
  GlobalHeaderBrand,
  GlobalHeaderInner,
  Select,
  SideNav,
  SideNavGroup,
  SideNavGroupLabel,
  SideNavItem,
  SideNavLink,
  SideNavList,
  SkipLink,
  StatusIndicator,
  Toolbar,
  ToolbarGroup,
  ToolbarItem,
  ToolbarLabel,
} from "lumen-ui-kit";
import { CloseIcon, Icon, MenuIcon, UserIcon } from "lumen-ui-kit/icons";
import {
  fetchBrainsList,
  getSession,
  mergeBrainOptions,
  setSession,
  type BrainRecord,
} from "../lib/api";
import { clearSession, loadSession, saveSession } from "../lib/auth";

const navGroups: {
  label: string;
  items: { to: string; label: string; end?: boolean }[];
}[] = [
  {
    label: "Explore",
    items: [
      { to: "/", label: "Overview", end: true },
      { to: "/graph", label: "Graph" },
    ],
  },
  {
    label: "Memory",
    items: [
      { to: "/data", label: "Data" },
      { to: "/observations", label: "Observations" },
      { to: "/vectors", label: "Vectors" },
    ],
  },
  {
    label: "Operations",
    items: [
      { to: "/tasks", label: "Tasks" },
      { to: "/ingest", label: "Ingest" },
      { to: "/sources", label: "Sources" },
    ],
  },
];

function isNavCurrent(pathname: string, to: string, end?: boolean) {
  if (end) return pathname === to;
  return pathname === to || pathname.startsWith(`${to}/`);
}

function BrainSwitcher({
  id,
  canSwitchBrains,
  brainId,
  brainOptions,
  loadingBrains,
  brainsError,
  onSwitch,
  onRetry,
  compact = false,
}: {
  id: string;
  canSwitchBrains: boolean;
  brainId: string;
  brainOptions: BrainRecord[];
  loadingBrains: boolean;
  brainsError: string | null;
  onSwitch: (id: string) => void;
  onRetry: () => void;
  compact?: boolean;
}) {
  if (compact) {
    return (
      <Toolbar
        aria-label="Brain workspace"
        density="compact"
        variant="plain"
        className="items-center"
      >
        <ToolbarLabel id={`${id}-label`}>Brain</ToolbarLabel>
        {canSwitchBrains ? (
          <ToolbarGroup aria-labelledby={`${id}-label`} className="items-center">
            <ToolbarItem className="flex items-center">
              <Select
                id={id}
                variant="ghost"
                value={brainId}
                onChange={(e) => onSwitch(e.target.value)}
                disabled={loadingBrains}
                className="h-9 w-auto min-w-[8rem] max-w-[14rem]"
              >
                {brainOptions.map((b) => (
                  <option key={b.name_key} value={b.name_key}>
                    {b.name_key}
                  </option>
                ))}
              </Select>
            </ToolbarItem>
            {loadingBrains ? (
              <ToolbarItem>
                <span className="px-1 text-xs text-lumen-muted-foreground">
                  Loading…
                </span>
              </ToolbarItem>
            ) : null}
            {brainsError ? (
              <ToolbarItem>
                <Button
                  type="button"
                  variant="secondary"
                  size="small"
                  onClick={onRetry}
                >
                  Retry
                </Button>
              </ToolbarItem>
            ) : null}
          </ToolbarGroup>
        ) : (
          <ToolbarGroup aria-labelledby={`${id}-label`} className="items-center">
            <ToolbarItem>
              <span className="px-1 text-sm font-medium text-lumen-foreground">
                {brainId}
              </span>
            </ToolbarItem>
          </ToolbarGroup>
        )}
      </Toolbar>
    );
  }

  if (!canSwitchBrains) {
    return (
      <div className="text-sm">
        <div className="text-lumen-muted-foreground">Active brain</div>
        <div className="font-medium text-lumen-foreground">{brainId}</div>
        <div className="mt-1 text-xs text-lumen-muted-foreground">
          Scoped PAT — this brain only
        </div>
      </div>
    );
  }

  return (
    <Field>
      <FieldLabel htmlFor={id}>Brain</FieldLabel>
      <Select
        id={id}
        value={brainId}
        onChange={(e) => onSwitch(e.target.value)}
        disabled={loadingBrains}
      >
        {brainOptions.map((b) => (
          <option key={b.name_key} value={b.name_key}>
            {b.name_key}
          </option>
        ))}
      </Select>
      {loadingBrains && <FieldDescription>Loading brains…</FieldDescription>}
      {brainsError && (
        <Button
          type="button"
          variant="secondary"
          size="small"
          onClick={onRetry}
          className="mt-1"
        >
          Retry brain list
        </Button>
      )}
    </Field>
  );
}

function ConsoleSideNav({
  pathname,
  onNavigate,
}: {
  pathname: string;
  onNavigate?: () => void;
}) {
  const navigate = useNavigate();

  return (
    <SideNav expression="compact" aria-label="Console sections">
      <SideNavList>
        {navGroups.map((group) => (
          <SideNavGroup key={group.label}>
            <SideNavGroupLabel>{group.label}</SideNavGroupLabel>
            <SideNavList>
              {group.items.map((item) => {
                const current = isNavCurrent(pathname, item.to, item.end);
                const href = `/console${item.to === "/" ? "/" : item.to}`;
                return (
                  <SideNavItem key={item.to}>
                    <SideNavLink
                      href={href}
                      current={current}
                      onClick={(e) => {
                        e.preventDefault();
                        navigate(item.to);
                        onNavigate?.();
                      }}
                    >
                      {item.label}
                    </SideNavLink>
                  </SideNavItem>
                );
              })}
            </SideNavList>
          </SideNavGroup>
        ))}
      </SideNavList>
    </SideNav>
  );
}

function SessionRail({
  brainId,
  apiBaseUrl,
  canSwitchBrains,
  isSystemPat,
}: {
  brainId: string;
  apiBaseUrl: string;
  canSwitchBrains: boolean;
  isSystemPat: boolean;
}) {
  return (
    <div className="flex flex-col gap-4 p-4">
      <div>
        <h2 className="text-sm font-semibold text-lumen-foreground">
          Session
        </h2>
        <p className="mt-1 text-xs text-lumen-muted-foreground">
          Connection and workspace context for this console.
        </p>
      </div>
      <DescriptionList className="gap-3 text-sm">
        <DescriptionTerm>Brain</DescriptionTerm>
        <DescriptionDetails className="font-mono text-xs">
          {brainId}
        </DescriptionDetails>
        <DescriptionTerm>API</DescriptionTerm>
        <DescriptionDetails className="break-all font-mono text-xs">
          {apiBaseUrl}
        </DescriptionDetails>
        <DescriptionTerm>Credentials</DescriptionTerm>
        <DescriptionDetails>
          {isSystemPat ? "System PAT" : "Per-brain PAT"}
        </DescriptionDetails>
        <DescriptionTerm>Status</DescriptionTerm>
        <DescriptionDetails>
          <StatusIndicator status="success">Connected</StatusIndicator>
        </DescriptionDetails>
        <DescriptionTerm>Switching</DescriptionTerm>
        <DescriptionDetails>
          {canSwitchBrains ? "Enabled" : "Locked to brain"}
        </DescriptionDetails>
      </DescriptionList>
    </div>
  );
}

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const session = getSession();
  const [brains, setBrains] = useState<BrainRecord[]>([]);
  const [brainId, setBrainId] = useState(session?.brainId ?? "default");
  const [canSwitchBrains, setCanSwitchBrains] = useState(
    session?.isSystemPat ?? false,
  );
  const [loadingBrains, setLoadingBrains] = useState(false);
  const [brainsError, setBrainsError] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const loadBrains = useCallback(async () => {
    const current = getSession();
    if (!current?.isSystemPat) {
      setCanSwitchBrains(false);
      setBrains([]);
      return;
    }

    setCanSwitchBrains(true);
    setLoadingBrains(true);
    setBrainsError(null);
    try {
      const list = await fetchBrainsList(current);
      setBrains(mergeBrainOptions(list, current.brainId));
    } catch (err) {
      setBrainsError(
        err instanceof Error ? err.message : "Failed to load brains",
      );
      setBrains(mergeBrainOptions([], current.brainId));
    } finally {
      setLoadingBrains(false);
    }
  }, []);

  useEffect(() => {
    setBrainId(session?.brainId ?? "default");
    setCanSwitchBrains(session?.isSystemPat ?? false);
    loadBrains();
  }, [loadBrains, session?.brainId, session?.isSystemPat]);

  function switchBrain(id: string) {
    setBrainId(id);
    const current = loadSession();
    if (current) {
      const next = { ...current, brainId: id };
      saveSession(next);
      setSession(next);
      window.location.reload();
    }
  }

  function logout() {
    clearSession();
    setSession(null);
    navigate("/login");
  }

  const brainOptions = mergeBrainOptions(brains, brainId);
  const pathname = location.pathname;
  const apiBaseUrl = session?.apiBaseUrl ?? "—";
  const isSystemPat = session?.isSystemPat ?? false;

  const headerBrain = (
    <BrainSwitcher
      id="header-brain-select"
      canSwitchBrains={canSwitchBrains}
      brainId={brainId}
      brainOptions={brainOptions}
      loadingBrains={loadingBrains}
      brainsError={brainsError}
      onSwitch={switchBrain}
      onRetry={loadBrains}
      compact
    />
  );

  const drawerBrain = (
    <BrainSwitcher
      id="drawer-brain-select"
      canSwitchBrains={canSwitchBrains}
      brainId={brainId}
      brainOptions={brainOptions}
      loadingBrains={loadingBrains}
      brainsError={brainsError}
      onSwitch={switchBrain}
      onRetry={loadBrains}
      compact
    />
  );

  return (
    <div className="flex min-h-screen flex-col bg-lumen-background">
      <SkipLink href="#console-main">Skip to main content</SkipLink>

      <GlobalHeader sticky className="border-b border-lumen-border">
        <GlobalHeaderInner className="max-w-none">
          <div className="lg:hidden">
            <Drawer open={drawerOpen} onOpenChange={setDrawerOpen}>
              <DrawerTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label="Open console navigation"
                >
                  <Icon source={MenuIcon} />
                </Button>
              </DrawerTrigger>
              <DrawerContent
                side="left"
                className="w-[min(20rem,calc(100vw-2rem))] gap-0 p-0"
              >
                <AppShellSidebarHeader className="flex items-center justify-between gap-3 border-b border-lumen-border p-3">
                  <DrawerTitle className="font-semibold">Console</DrawerTitle>
                  <DrawerClose asChild>
                    <Button
                      size="icon"
                      variant="ghost"
                      aria-label="Close console navigation"
                    >
                      <Icon source={CloseIcon} />
                    </Button>
                  </DrawerClose>
                </AppShellSidebarHeader>
                <DrawerDescription className="sr-only">
                  Navigate BrainAPI Console sections.
                </DrawerDescription>
                <div className="border-b border-lumen-border p-3">
                  {drawerBrain}
                </div>
                <AppShellSidebarContent className="p-0">
                  <ConsoleSideNav
                    pathname={pathname}
                    onNavigate={() => setDrawerOpen(false)}
                  />
                </AppShellSidebarContent>
                <AppShellSidebarFooter className="mt-auto border-t border-lumen-border p-3">
                  <Button
                    type="button"
                    variant="secondary"
                    isFullWidth
                    onClick={logout}
                  >
                    Log out
                  </Button>
                </AppShellSidebarFooter>
              </DrawerContent>
            </Drawer>
          </div>
          <GlobalHeaderBrand href="/console/">
            <span
              aria-hidden="true"
              className="grid size-8 shrink-0 place-items-center bg-lumen-primary text-xs font-bold text-lumen-on-primary"
            >
              B
            </span>
            <span className="grid leading-tight">
              <span className="font-semibold">BrainAPI</span>
              <span className="text-xs font-normal text-lumen-muted-foreground">
                Operations console
              </span>
            </span>
          </GlobalHeaderBrand>
          <GlobalHeaderActions className="ml-auto flex-nowrap items-center gap-3">
            <div className="hidden md:flex md:items-center">{headerBrain}</div>
            <div className="hidden max-w-[14rem] truncate text-xs text-lumen-muted-foreground xl:block">
              {apiBaseUrl}
            </div>
            <StatusIndicator status="success" className="hidden sm:inline-flex">
              Live
            </StatusIndicator>
            <Button
              type="button"
              size="small"
              variant="secondary"
              onClick={logout}
            >
              <Icon source={UserIcon} />
              Log out
            </Button>
          </GlobalHeaderActions>
        </GlobalHeaderInner>
      </GlobalHeader>

      <AppShell
        layout="sidebar-rail"
        className="min-h-0 flex-1 overflow-hidden"
        style={{
          minHeight:
            "calc(100vh - var(--lumen-layout-header-height, 3.5rem))",
        }}
      >
        <AppShellSidebar className="hidden min-h-0 overflow-hidden lg:flex">
          <AppShellSidebarHeader className="border-b border-lumen-border font-normal">
            <p className="text-xs font-medium uppercase tracking-wide text-lumen-muted-foreground">
              Workspace
            </p>
            <p className="mt-1 font-mono text-sm text-lumen-foreground">
              {brainId}
            </p>
          </AppShellSidebarHeader>
          <AppShellSidebarContent className="min-h-0 overflow-auto">
            <ConsoleSideNav pathname={pathname} />
          </AppShellSidebarContent>
          <AppShellSidebarFooter className="border-t border-lumen-border text-xs text-lumen-muted-foreground">
            {isSystemPat ? "System credentials" : "Scoped credentials"}
          </AppShellSidebarFooter>
        </AppShellSidebar>

        <AppShellMain
          as="main"
          id="console-main"
          className="flex min-h-0 min-w-0 flex-col overflow-hidden"
        >
          <Outlet />
          <div className="border-t border-lumen-border px-4 pb-4 pt-3 xl:hidden">
            <Disclosure className="border border-lumen-border bg-lumen-surface">
              <DisclosureTrigger className="w-full px-4 py-3 text-left text-sm font-medium">
                Workspace context
              </DisclosureTrigger>
              <DisclosureContent>
                <SessionRail
                  brainId={brainId}
                  apiBaseUrl={apiBaseUrl}
                  canSwitchBrains={canSwitchBrains}
                  isSystemPat={isSystemPat}
                />
              </DisclosureContent>
            </Disclosure>
          </div>
        </AppShellMain>

        <AppShellRail
          className="hidden min-h-0 overflow-auto xl:flex"
          aria-label="Workspace context"
        >
          <SessionRail
            brainId={brainId}
            apiBaseUrl={apiBaseUrl}
            canSwitchBrains={canSwitchBrains}
            isSystemPat={isSystemPat}
          />
        </AppShellRail>
      </AppShell>
    </div>
  );
}
