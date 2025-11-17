// Firebase Authentication Service
// Configuration loaded from firebase-config.js

class FirebaseAuthService {
    constructor() {
        this.auth = null;
        this.currentUser = null;
        this.initialized = false;
        this.tokenRefreshTimer = null;
        this.activityCheckTimer = null;
        this.activityListenersSetup = false;
        this.isRefreshingToken = false;
    }

    // Initialize Firebase
    async initialize() {
        try {
            // Check if Firebase SDK is loaded
            if (typeof firebase === 'undefined' || !firebase.auth) {
                console.warn('Firebase SDK not loaded yet');
                this.hideAuthButtons();
                return false;
            }

            // Check if config is available
            if (!window.firebaseConfig) {
                console.warn('Firebase config not found. Please update firebase-config.js with your credentials.');
                this.hideAuthButtons();
                return false;
            }

            // Check if already initialized
            if (!this.initialized) {
                // Initialize Firebase app
                if (!firebase.apps.length) {
                    firebase.initializeApp(window.firebaseConfig);
                }

                this.auth = firebase.auth();
                this.initialized = true;

                // Set up auth state observer
                this.auth.onAuthStateChanged((user) => {
                    this.currentUser = user;
                    this.onAuthStateChanged(user);
                });

                // token refresh
                this.auth.onIdTokenChanged(async (user) => {
                    if (user) {
                        const token = await user.getIdToken();
                        if (window.apiService) {
                        window.apiService.setAuthToken(token);
                        }
                        console.log('Firebase ID token refreshed automatically');
                    }
                });

                console.log('Firebase initialized successfully');
            }

            return true;
        } catch (error) {
            console.error('Firebase initialization error:', error);

            // Show user-friendly error message
            if (error.code === 'auth/invalid-api-key') {
                console.error('Invalid Firebase API key. Please update firebase-config.js with valid credentials.');
            } else if (error.code === 'auth/configuration-not-found') {
                console.error('Firebase Authentication is not enabled. Please enable it in Firebase Console.');
                this.showFirebaseSetupError();
            }

            this.hideAuthButtons();
            return false;
        }
    }

    startTokenRefreshTimer(user) {
        // clearInterval
        if (this.tokenRefreshTimer) {
            clearInterval(this.tokenRefreshTimer);
            this.tokenRefreshTimer = null;
        }

        if (!user) return;

        // 50분마다 갱신 (토큰 만료는 1시간이므로 여유있게)
        const intervalMs = 50 * 60 * 1000;

        // 사용자 활동 감지 리스너 추가
        this.setupActivityListeners();

        this.tokenRefreshTimer = setInterval(async () => {
            await this.refreshToken(user, 'periodic timer');
        }, intervalMs);
    }

    // 토큰 갱신 공통 함수 (중복 방지 포함)
    async refreshToken(user, source = 'unknown') {
        if (this.isRefreshingToken) {
            console.log(`[FirebaseAuth] Token refresh already in progress (${source})`);
            return false;
        }

        this.isRefreshingToken = true;

        try {
            console.log(`[FirebaseAuth] Refreshing ID token (${source})...`);
            const newToken = await user.getIdToken(true);  // 강제 새 토큰
            if (window.apiService) {
                window.apiService.setAuthToken(newToken);
            }
            console.log(`[FirebaseAuth] ID token refreshed by ${source}`);
            return true;
        } catch (err) {
            console.error(`[FirebaseAuth] Token refresh error (${source}):`, err);
            return false;
        } finally {
            this.isRefreshingToken = false;
        }
    }

    // 사용자 활동 감지 및 토큰 갱신
    setupActivityListeners() {
        if (this.activityListenersSetup) return;
        
        let lastActivity = Date.now();

        const updateActivity = () => {
            lastActivity = Date.now();
        };

        // 사용자 활동 감지 (클릭, 키보드, 마우스 이동, 스크롤)
        ['click', 'keydown', 'mousemove', 'scroll'].forEach(event => {
            document.addEventListener(event, updateActivity, { passive: true });
        });

        // 5분마다 활동 체크
        this.activityCheckTimer = setInterval(async () => {
            const minutesSinceActivity = (Date.now() - lastActivity) / (1000 * 60);
            const user = this.auth?.currentUser;
            
            // 최근 5분 내 활동이 있고 사용자가 로그인된 상태
            if (user && minutesSinceActivity < 5) {
                try {
                    const tokenResult = await user.getIdTokenResult();
                    const expirationTime = new Date(tokenResult.expirationTime).getTime();
                    const timeUntilExpiry = (expirationTime - Date.now()) / (1000 * 60);
                    
                    // 만료 15분 전에 갱신
                    if (timeUntilExpiry < 15) {
                        await this.refreshToken(user, 'activity check');
                    }
                } catch (err) {
                    console.error('[FirebaseAuth] Activity-based token check error:', err);
                }
            }
        }, 5 * 60 * 1000); // 5분마다 체크

        this.activityListenersSetup = true;
        console.log('[FirebaseAuth] Activity listeners setup complete');
    }

    // timer clear
    clearTokenRefreshTimer() {
        if (this.tokenRefreshTimer) {
            clearInterval(this.tokenRefreshTimer);
            this.tokenRefreshTimer = null;
        }
        if (this.activityCheckTimer) {
            clearInterval(this.activityCheckTimer);
            this.activityCheckTimer = null;
        }
        this.activityListenersSetup = false;
    }

    // Hide auth buttons when Firebase is not configured
    hideAuthButtons() {
        const authButtons = document.getElementById('auth-buttons');
        if (authButtons) {
            authButtons.innerHTML = `
                <div class="alert alert-warning mb-0 py-2 px-3 small" role="alert">
                    <i class="bi bi-exclamation-triangle me-1"></i>
                    Authentication disabled (Firebase not configured)
                </div>
            `;
        }
    }

    // Show Firebase setup error
    showFirebaseSetupError() {
        const container = document.createElement('div');
        container.className = 'position-fixed top-0 start-50 translate-middle-x mt-3';
        container.style.zIndex = '9999';
        container.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show shadow-lg" role="alert">
                <h6 class="alert-heading mb-2">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    Firebase Authentication Not Configured
                </h6>
                <p class="mb-2 small">Please enable Email/Password authentication in Firebase Console:</p>
                <ol class="mb-2 small ps-3">
                    <li>Go to <a href="https://console.firebase.google.com/" target="_blank" class="alert-link">Firebase Console</a></li>
                    <li>Select your project: <strong>final-pjt-860a7</strong></li>
                    <li>Go to <strong>Authentication</strong> → <strong>Sign-in method</strong></li>
                    <li>Enable <strong>Email/Password</strong> provider</li>
                    <li>Optionally enable <strong>Google</strong> provider</li>
                </ol>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        document.body.appendChild(container);

        // Auto remove after 15 seconds
        setTimeout(() => {
            if (container.parentNode) {
                container.remove();
            }
        }, 15000);
    }

    // Auth state change handler
    onAuthStateChanged(user) {
        if (user) {
            // User is signed in
            console.log('User signed in:', user.email);

            // After signed in
            this.startTokenRefreshTimer(user);

            // Get Firebase ID token and set it in API service
            user.getIdToken().then((token) => {
                if (window.apiService) {
                    window.apiService.setAuthToken(token);
                }

                // Store user info
                localStorage.setItem('user', JSON.stringify({
                    uid: user.uid,
                    email: user.email,
                    displayName: user.displayName,
                    photoURL: user.photoURL
                }));

                // Update UI
                this.updateAuthUI(user);
            });
        } else {
            // User is signed out
            console.log('User signed out');

            if (window.apiService) {
                window.apiService.clearAuth();
            }

            // after signed out
            this.clearTokenRefreshTimer();

            localStorage.removeItem('user');
            this.updateAuthUI(null);
        }
    }

    // Update authentication UI elements
    updateAuthUI(user) {
        const authButtons = document.getElementById('auth-buttons');
        const userProfile = document.getElementById('user-profile');

        if (!authButtons || !userProfile) return;

        if (user) {
            // User is logged in
            authButtons.classList.add('d-none');
            userProfile.classList.remove('d-none');
            userProfile.classList.add('d-flex');

            const userNameElement = document.getElementById('user-name');
            const userPhotoElement = document.getElementById('user-photo');

            if (userNameElement) {
                userNameElement.textContent = user.displayName || user.email;
            }

            if (userPhotoElement && user.photoURL) {
                userPhotoElement.src = user.photoURL;
            }
        } else {
            // User is logged out
            authButtons.classList.remove('d-none');
            authButtons.classList.add('d-flex');
            userProfile.classList.add('d-none');
            userProfile.classList.remove('d-flex');
        }
    }

    // Sign in with email and password
    async signInWithEmail(email, password) {
        try {
            const userCredential = await this.auth.signInWithEmailAndPassword(email, password);
            return userCredential.user;
        } catch (error) {
            console.error('Sign in error:', error);
            throw this.handleAuthError(error);
        }
    }

    // Sign up with email and password
    async signUpWithEmail(email, password, displayName) {
        try {
            const userCredential = await this.auth.createUserWithEmailAndPassword(email, password);

            // Update profile with display name
            if (displayName) {
                await userCredential.user.updateProfile({
                    displayName: displayName
                });
            }

            return userCredential.user;
        } catch (error) {
            console.error('Sign up error:', error);
            throw this.handleAuthError(error);
        }
    }

    // Sign in with Google
    async signInWithGoogle() {
        try {
            const provider = new firebase.auth.GoogleAuthProvider();
            const userCredential = await this.auth.signInWithPopup(provider);
            return userCredential.user;
        } catch (error) {
            console.error('Google sign in error:', error);
            throw this.handleAuthError(error);
        }
    }

    // Sign out
    async signOut() {
        try {
            await this.auth.signOut();
            this.clearTokenRefreshTimer();
            // Redirect to home page
            window.location.hash = '#/';
        } catch (error) {
            console.error('Sign out error:', error);
            throw error;
        }
    }

    // Get current user
    getCurrentUser() {
        return this.currentUser;
    }

    // Check if user is authenticated
    isAuthenticated() {
        return this.currentUser !== null;
    }

    // Get ID token for API calls
    async getIdToken() {
        if (this.currentUser) {
            return await this.currentUser.getIdToken();
        }
        return null;
    }

    // Handle authentication errors
    handleAuthError(error) {
        const errorMessages = {
            'auth/email-already-in-use': 'This email is already registered',
            'auth/invalid-email': 'Invalid email address',
            'auth/operation-not-allowed': 'Operation not allowed',
            'auth/weak-password': 'Password is too weak',
            'auth/user-disabled': 'This account has been disabled',
            'auth/user-not-found': 'No account found with this email',
            'auth/wrong-password': 'Incorrect password',
            'auth/popup-closed-by-user': 'Sign in popup was closed',
            'auth/cancelled-popup-request': 'Sign in cancelled'
        };

        return new Error(errorMessages[error.code] || error.message);
    }

    // Send password reset email
    async sendPasswordResetEmail(email) {
        try {
            await this.auth.sendPasswordResetEmail(email);
        } catch (error) {
            console.error('Password reset error:', error);
            throw this.handleAuthError(error);
        }
    }

    // Update user profile
    async updateProfile(displayName, photoURL) {
        try {
            if (this.currentUser) {
                const updates = {};
                if (displayName) updates.displayName = displayName;
                if (photoURL) updates.photoURL = photoURL;

                await this.currentUser.updateProfile(updates);
            }
        } catch (error) {
            console.error('Profile update error:', error);
            throw error;
        }
    }
}

// Create singleton instance
const firebaseAuthService = new FirebaseAuthService();

// Show login modal
function showLoginModal() {
    const modalHTML = `
        <div class="modal fade" id="loginModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Sign In</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div id="login-error" class="alert alert-danger d-none"></div>

                        <form id="login-form">
                            <div class="mb-3">
                                <label for="login-email" class="form-label">Email</label>
                                <input type="email" class="form-control" id="login-email" required>
                            </div>
                            <div class="mb-3">
                                <label for="login-password" class="form-label">Password</label>
                                <input type="password" class="form-control" id="login-password" required>
                            </div>
                            <div class="mb-3">
                                <a href="#" id="forgot-password-link">Forgot password?</a>
                            </div>
                            <button type="submit" class="btn btn-primary w-100">Sign In</button>
                        </form>

                        <div class="text-center my-3">
                            <span class="text-muted">or</span>
                        </div>

                        <button id="google-signin-btn" class="btn btn-outline-dark w-100">
                            <i class="bi bi-google me-2"></i>Sign in with Google
                        </button>

                        <div class="text-center mt-3">
                            <span class="text-muted">Don't have an account? </span>
                            <a href="#" id="show-signup-link">Sign up</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if any
    const existingModal = document.getElementById('loginModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHTML);

    // Get modal element
    const modalElement = document.getElementById('loginModal');
    const modal = new bootstrap.Modal(modalElement);

    // Handle form submission
    document.getElementById('login-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
        const errorDiv = document.getElementById('login-error');

        try {
            await firebaseAuthService.signInWithEmail(email, password);
            modal.hide();
            showToast('Sign in successful!', 'success');
        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.classList.remove('d-none');
        }
    });

    // Handle Google sign in
    document.getElementById('google-signin-btn').addEventListener('click', async () => {
        const errorDiv = document.getElementById('login-error');

        try {
            await firebaseAuthService.signInWithGoogle();
            modal.hide();
            showToast('Sign in successful!', 'success');
        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.classList.remove('d-none');
        }
    });

    // Handle show signup
    document.getElementById('show-signup-link').addEventListener('click', (e) => {
        e.preventDefault();
        modal.hide();
        showSignupModal();
    });

    // Handle forgot password
    document.getElementById('forgot-password-link').addEventListener('click', (e) => {
        e.preventDefault();
        modal.hide();
        showForgotPasswordModal();
    });

    modal.show();
}

// Show signup modal
function showSignupModal() {
    const modalHTML = `
        <div class="modal fade" id="signupModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Sign Up</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div id="signup-error" class="alert alert-danger d-none"></div>

                        <form id="signup-form">
                            <div class="mb-3">
                                <label for="signup-name" class="form-label">Display Name</label>
                                <input type="text" class="form-control" id="signup-name" required>
                            </div>
                            <div class="mb-3">
                                <label for="signup-email" class="form-label">Email</label>
                                <input type="email" class="form-control" id="signup-email" required>
                            </div>
                            <div class="mb-3">
                                <label for="signup-password" class="form-label">Password</label>
                                <input type="password" class="form-control" id="signup-password" required minlength="6">
                                <small class="text-muted">At least 6 characters</small>
                            </div>
                            <div class="mb-3">
                                <label for="signup-password-confirm" class="form-label">Confirm Password</label>
                                <input type="password" class="form-control" id="signup-password-confirm" required>
                            </div>
                            <button type="submit" class="btn btn-primary w-100">Sign Up</button>
                        </form>

                        <div class="text-center mt-3">
                            <span class="text-muted">Already have an account? </span>
                            <a href="#" id="show-login-link">Sign in</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    const existingModal = document.getElementById('signupModal');
    if (existingModal) {
        existingModal.remove();
    }

    document.body.insertAdjacentHTML('beforeend', modalHTML);

    const modalElement = document.getElementById('signupModal');
    const modal = new bootstrap.Modal(modalElement);

    document.getElementById('signup-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const name = document.getElementById('signup-name').value;
        const email = document.getElementById('signup-email').value;
        const password = document.getElementById('signup-password').value;
        const passwordConfirm = document.getElementById('signup-password-confirm').value;
        const errorDiv = document.getElementById('signup-error');

        if (password !== passwordConfirm) {
            errorDiv.textContent = 'Passwords do not match';
            errorDiv.classList.remove('d-none');
            return;
        }

        try {
            await firebaseAuthService.signUpWithEmail(email, password, name);
            modal.hide();
            showToast('Account created successfully!', 'success');
        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.classList.remove('d-none');
        }
    });

    document.getElementById('show-login-link').addEventListener('click', (e) => {
        e.preventDefault();
        modal.hide();
        showLoginModal();
    });

    modal.show();
}

// Show forgot password modal
function showForgotPasswordModal() {
    const modalHTML = `
        <div class="modal fade" id="forgotPasswordModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Reset Password</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div id="forgot-error" class="alert alert-danger d-none"></div>
                        <div id="forgot-success" class="alert alert-success d-none"></div>

                        <form id="forgot-password-form">
                            <div class="mb-3">
                                <label for="forgot-email" class="form-label">Email</label>
                                <input type="email" class="form-control" id="forgot-email" required>
                                <small class="text-muted">We'll send you a password reset link</small>
                            </div>
                            <button type="submit" class="btn btn-primary w-100">Send Reset Link</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    `;

    const existingModal = document.getElementById('forgotPasswordModal');
    if (existingModal) {
        existingModal.remove();
    }

    document.body.insertAdjacentHTML('beforeend', modalHTML);

    const modalElement = document.getElementById('forgotPasswordModal');
    const modal = new bootstrap.Modal(modalElement);

    document.getElementById('forgot-password-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const email = document.getElementById('forgot-email').value;
        const errorDiv = document.getElementById('forgot-error');
        const successDiv = document.getElementById('forgot-success');

        try {
            await firebaseAuthService.sendPasswordResetEmail(email);
            successDiv.textContent = 'Password reset email sent! Check your inbox.';
            successDiv.classList.remove('d-none');
            errorDiv.classList.add('d-none');

            setTimeout(() => {
                modal.hide();
            }, 2000);
        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.classList.remove('d-none');
            successDiv.classList.add('d-none');
        }
    });

    modal.show();
}

// Show toast notification
function showToast(message, type = 'info') {
    const toastHTML = `
        <div class="toast align-items-center text-white bg-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }

    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement);
    toast.show();

    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

// Initialize Firebase when DOM is ready
window.addEventListener('load', () => {
    // Wait a bit for all scripts to load
    setTimeout(() => {
        firebaseAuthService.initialize().then(success => {
            if (!success) {
                console.warn('Firebase not initialized. Auth features will be disabled.');
            }
        }).catch(error => {
            console.error('Firebase initialization failed:', error);
        });
    }, 100);
});
