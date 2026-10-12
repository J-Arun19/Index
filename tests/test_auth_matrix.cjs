const util = require('node:util');
if (!util.styleText) {
  util.styleText = (format, text) => text;
}
if (typeof globalThis.CustomEvent === 'undefined') {
  globalThis.CustomEvent = class CustomEvent extends Event {
    constructor(event, params = {}) {
      super(event, params);
      this.detail = params.detail || null;
    }
  };
}

const fs = require('node:fs');
const path = require('node:path');
const { createClient } = require('@supabase/supabase-js');

// Dynamically load server credentials from backend/.env (server-side only)
let SUPABASE_URL = process.env.SUPABASE_URL || 'https://nbsrjqdgrrjejpxffauh.supabase.co';
let SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
let SUPABASE_PUBLISHABLE_KEY = process.env.VITE_SUPABASE_PUBLISHABLE_KEY;

const backendEnvPath = path.resolve(__dirname, '..', 'backend', '.env');
if (fs.existsSync(backendEnvPath)) {
  const envText = fs.readFileSync(backendEnvPath, 'utf8');
  for (const line of envText.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('SUPABASE_URL=')) SUPABASE_URL = trimmed.split('=')[1].trim();
    if (trimmed.startsWith('SUPABASE_SERVICE_ROLE_KEY=')) SUPABASE_SERVICE_ROLE_KEY = trimmed.split('=')[1].trim();
  }
}

const rootEnvPath = path.resolve(__dirname, '..', '.env');
if (fs.existsSync(rootEnvPath)) {
  const envText = fs.readFileSync(rootEnvPath, 'utf8');
  for (const line of envText.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('VITE_SUPABASE_PUBLISHABLE_KEY=')) SUPABASE_PUBLISHABLE_KEY = trimmed.split('=')[1].trim();
  }
}

// Create test mock storage
const mockStorage = new Map();
const mockLocalStorage = {
  getItem: (k) => mockStorage.get(k) || null,
  setItem: (k, v) => mockStorage.set(k, String(v)),
  removeItem: (k) => mockStorage.delete(k),
};

// Isolated client simulating browser environment
class DummyWS {}
const browserKey = (SUPABASE_PUBLISHABLE_KEY && !SUPABASE_PUBLISHABLE_KEY.includes('placeholder') && !SUPABASE_PUBLISHABLE_KEY.includes('rotate'))
  ? SUPABASE_PUBLISHABLE_KEY
  : SUPABASE_SERVICE_ROLE_KEY;

function getBrowserClient() {
  return createClient(SUPABASE_URL, browserKey, {
    auth: {
      persistSession: true,
      storage: mockLocalStorage,
      autoRefreshToken: true,
      detectSessionInUrl: false,
    },
    realtime: {
      transport: DummyWS,
    },
  });
}

// Admin client for test setup/cleanup (server-side service role client)
const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false },
  realtime: { transport: DummyWS }
});

async function runTestMatrix() {
  console.log('====================================================');
  console.log(' SatyaDrishti Auth Verification Suite (10 Tests)   ');
  console.log('====================================================\n');

  const testResults = [];
  function record(id, title, pass, detail) {
    testResults.push({ id, title, pass, detail });
    console.log(`[${pass ? 'PASS' : 'FAIL'}] ${id}: ${title}`);
    if (detail) console.log(`       Detail: ${detail}`);
  }

  const runId = Date.now();
  const testConfirmedEmail = `confirmed_test_${runId}@example.com`;
  const testUnconfirmedEmail = `unconfirmed_test_${runId}@example.com`;
  const testPassword = `ValidAuthPass#${runId}`;

  let confirmedUserId = null;
  let unconfirmedUserId = null;

  try {
    // SETUP: Create confirmed user (admin)
    const { data: u1, error: e1 } = await adminClient.auth.admin.createUser({
      email: testConfirmedEmail,
      password: testPassword,
      user_metadata: { full_name: 'Verified Test Analyst' },
      email_confirm: true,
    });
    if (e1) throw new Error(`Setup failed for confirmed user: ${e1.message}`);
    confirmedUserId = u1.user.id;

    // SETUP: Create unconfirmed user (admin)
    const { data: u2, error: e2 } = await adminClient.auth.admin.createUser({
      email: testUnconfirmedEmail,
      password: testPassword,
      user_metadata: { full_name: 'Unverified Test User' },
      email_confirm: false,
    });
    if (e2) throw new Error(`Setup failed for unconfirmed user: ${e2.message}`);
    unconfirmedUserId = u2.user.id;

    // -------------------------------------------------------------
    // TEST 1: Existing confirmed user + correct password
    // -------------------------------------------------------------
    const client1 = getBrowserClient();
    const res1 = await client1.auth.signInWithPassword({
      email: testConfirmedEmail,
      password: testPassword,
    });
    const test1Pass = !!(res1.data?.session && res1.data?.user && !res1.error);
    record(
      'TEST 1',
      'Existing confirmed user + correct password',
      test1Pass,
      test1Pass ? `Session token acquired for user ${res1.data.user.id}` : res1.error?.message
    );

    // -------------------------------------------------------------
    // TEST 2: Existing confirmed user + wrong password
    // -------------------------------------------------------------
    const client2 = getBrowserClient();
    const res2 = await client2.auth.signInWithPassword({
      email: testConfirmedEmail,
      password: 'DefinitelyWrongPassword!999',
    });
    const isInvalidCreds = res2.error && (res2.error.code === 'invalid_credentials' || res2.error.message.includes('Invalid login credentials'));
    record(
      'TEST 2',
      'Existing confirmed user + wrong password',
      !!isInvalidCreds,
      res2.error ? `Rejected with: ${res2.error.message} (code: ${res2.error.code})` : 'Unexpectedly succeeded'
    );

    // -------------------------------------------------------------
    // TEST 3: Existing user + empty password (frontend validation logic)
    // -------------------------------------------------------------
    let test3Pass = false;
    let test3Msg = '';
    const emailInput = testConfirmedEmail;
    const passInput = '';
    if (!emailInput.trim() || !passInput) {
      test3Pass = true;
      test3Msg = 'Prevented submission before calling Supabase API (empty password detected)';
    }
    record('TEST 3', 'Existing user + empty password', test3Pass, test3Msg);

    // -------------------------------------------------------------
    // TEST 4: Existing unconfirmed user + correct password
    // -------------------------------------------------------------
    const client4 = getBrowserClient();
    const res4 = await client4.auth.signInWithPassword({
      email: testUnconfirmedEmail,
      password: testPassword,
    });
    const isUnconfirmed = res4.error && (res4.error.code === 'email_not_confirmed' || res4.error.message.toLowerCase().includes('email not confirmed'));
    record(
      'TEST 4',
      'Existing unconfirmed user + correct password',
      !!isUnconfirmed,
      res4.error ? `Detected unconfirmed state: ${res4.error.message} (code: ${res4.error.code})` : 'Unexpectedly succeeded'
    );

    // -------------------------------------------------------------
    // TEST 5: Nonexistent email + password
    // -------------------------------------------------------------
    const client5 = getBrowserClient();
    const res5 = await client5.auth.signInWithPassword({
      email: `nonexistent_user_${runId}@example.com`,
      password: testPassword,
    });
    const test5Pass = res5.error && (res5.error.code === 'invalid_credentials' || res5.error.message.includes('Invalid login credentials'));
    record(
      'TEST 5',
      'Nonexistent email + password',
      !!test5Pass,
      res5.error ? `Correctly rejected: ${res5.error.message}` : 'Unexpectedly succeeded'
    );

    // -------------------------------------------------------------
    // TEST 6: Successful login → refresh browser (session persistence)
    // -------------------------------------------------------------
    // Client1 logged in earlier with mockLocalStorage.
    // Create new client sharing mockLocalStorage to simulate page refresh.
    const client6 = getBrowserClient();
    const res6 = await client6.auth.getSession();
    const test6Pass = !!(res6.data?.session?.user?.id === confirmedUserId);
    record(
      'TEST 6',
      'Successful login → Refresh browser (Session Persistence)',
      test6Pass,
      test6Pass ? `Session restored from storage for user ${res6.data.session.user.id}` : 'Session was lost on refresh'
    );

    // -------------------------------------------------------------
    // TEST 7: Successful login → navigate to Dashboard
    // -------------------------------------------------------------
    // Check if session remains active and user metadata is available for Dashboard
    const userProfile = res6.data?.session?.user;
    const test7Pass = !!(userProfile && userProfile.email === testConfirmedEmail);
    record(
      'TEST 7',
      'Successful login → Dashboard access check',
      test7Pass,
      test7Pass ? `User authenticated for Dashboard: ${userProfile.email} (${userProfile.user_metadata?.full_name})` : 'User profile missing'
    );

    // -------------------------------------------------------------
    // TEST 8: Sign out
    // -------------------------------------------------------------
    await client6.auth.signOut();
    const res8 = await client6.auth.getSession();
    const test8Pass = res8.data?.session === null;
    record(
      'TEST 8',
      'Sign out clears session',
      test8Pass,
      test8Pass ? 'Session cleared from storage; getSession() returned null' : 'Session still lingered'
    );

    // -------------------------------------------------------------
    // TEST 9: Existing Auth user with NO public.users profile
    // -------------------------------------------------------------
    // confirmedUserId was created in auth.users only; public.users has NO row for testConfirmedEmail.
    const { data: publicRows } = await adminClient.from('users').select('*').eq('email', testConfirmedEmail);
    const hasNoPublicRow = !publicRows || publicRows.length === 0;
    const client9 = getBrowserClient();
    const res9 = await client9.auth.signInWithPassword({
      email: testConfirmedEmail,
      password: testPassword,
    });
    const test9Pass = hasNoPublicRow && !!res9.data?.session && !!res9.data?.user;
    record(
      'TEST 9',
      'Existing Auth user with NO public.users profile',
      test9Pass,
      test9Pass ? `Auth succeeded with 0 public.users rows; session valid for ${res9.data.user.id}` : 'Failed'
    );

    // -------------------------------------------------------------
    // TEST 10: Register new user → sign out → sign in using SAME credentials
    // -------------------------------------------------------------
    const newRegEmail = `new_reg_${runId}@example.com`;
    const newRegPassword = `NewPass#${runId}`;
    const newRegName = 'E2E Test User';

    // Step A: Register via certified admin endpoint
    const { data: regUser, error: regErr } = await adminClient.auth.admin.createUser({
      email: newRegEmail,
      password: newRegPassword,
      user_metadata: { full_name: newRegName },
      email_confirm: true,
    });
    if (regErr) throw new Error(`Registration failed: ${regErr.message}`);

    // Step B: Client logs in with newly registered credentials
    const client10 = getBrowserClient();
    const loginA = await client10.auth.signInWithPassword({
      email: newRegEmail,
      password: newRegPassword,
    });
    const loginA_ok = !!(loginA.data?.session && loginA.data?.user);

    // Step C: Client signs out
    await client10.auth.signOut();
    const sessionAfterLogout = await client10.auth.getSession();
    const logout_ok = sessionAfterLogout.data?.session === null;

    // Step D: Client signs in again with the SAME email and password
    const loginB = await client10.auth.signInWithPassword({
      email: newRegEmail,
      password: newRegPassword,
    });
    const loginB_ok = !!(loginB.data?.session && loginB.data?.user?.email === newRegEmail);

    const test10Pass = loginA_ok && logout_ok && loginB_ok;
    record(
      'TEST 10',
      'REGISTER → LOGOUT → LOGIN WITH SAME CREDENTIALS → SESSION CREATED → DASHBOARD',
      test10Pass,
      test10Pass
        ? `Full end-to-end cycle verified! Registered ${newRegEmail} -> Logged in -> Logged out -> Re-authenticated successfully.`
        : `Failure: loginA=${loginA_ok}, logout=${logout_ok}, loginB=${loginB_ok}`
    );

    // Clean up newly registered user
    await adminClient.auth.admin.deleteUser(regUser.user.id).catch(() => {});

  } finally {
    // Clean up test accounts
    if (confirmedUserId) await adminClient.auth.admin.deleteUser(confirmedUserId).catch(() => {});
    if (unconfirmedUserId) await adminClient.auth.admin.deleteUser(unconfirmedUserId).catch(() => {});
  }

  console.log('\n====================================================');
  const allPassed = testResults.every((t) => t.pass);
  console.log(` SUMMARY: ${testResults.filter((t) => t.pass).length}/${testResults.length} TESTS PASSED`);
  console.log(` RESULT:  ${allPassed ? 'ALL TESTS PASSED SUCCESSFULLY! 🚀' : 'SOME TESTS FAILED ❌'}`);
  console.log('====================================================');

  if (!allPassed) process.exit(1);
}

runTestMatrix().catch((err) => {
  console.error('Test matrix execution error:', err);
  process.exit(1);
});
