// VisionAI Platform - Main App Router

class App {
    constructor() {
        this.app = document.getElementById('app');
        this.routes = {
            '/': HomePage,
            '/datasets': DatasetsPage,
            '/export': ExportPage,
            '/training': TrainingPage,
            '/conversion': ConversionPage,
            '/evaluation': EvaluationPage,
            '/deployment': DeploymentPage,
        };

        // Dynamic routes (with parameters)
        this.dynamicRoutes = [
            { pattern: /^\/auto-annotate\/(\d+)$/, handler: (id) => new AutoAnnotatePage(id) },
            { pattern: /^\/dataset-detail\/(\d+)$/, handler: (id) => new DatasetDetailPage(id) },
            { pattern: /^\/upload-labels\/(\d+)$/, handler: (id) => new UploadLabelsPage(id) }
        ];

        this.init();
    }

    init() {
        // Listen for hash changes
        window.addEventListener('hashchange', () => this.router());

        // Listen for page load
        window.addEventListener('load', () => this.router());

        // Add navigation listeners
        this.setupNavigation();
    }

    router() {
        const path = window.location.hash.slice(1) || '/';

        // Check static routes first
        let page = this.routes[path];

        // If no static route found, check dynamic routes
        if (!page) {
            for (const route of this.dynamicRoutes) {
                const match = path.match(route.pattern);
                if (match) {
                    // Extract parameter and create page instance
                    const param = match[1];
                    page = route.handler(param);
                    break;
                }
            }
        }

        // If still no page found, use NotFoundPage
        if (!page) {
            page = NotFoundPage;
        }

        // Update active nav link
        this.updateActiveNav(path);

        // Render page
        this.render(page);
    }

    render(pageInstance) {
        // Cleanup previous page instance
        if (window.currentPageInstance && window.currentPageInstance.cleanup) {
            window.currentPageInstance.cleanup();
        }

        // Check if it's a class or an instance
        if (typeof pageInstance === 'function') {
            pageInstance = new pageInstance();
        }

        this.app.innerHTML = pageInstance.render();

        // Call init if exists (for async initialization)
        if (pageInstance.init) {
            pageInstance.init().catch(error => {
                console.error('Page initialization error:', error);
            });
        }

        // Call afterRender if exists
        if (pageInstance.afterRender) {
            setTimeout(() => pageInstance.afterRender(), 0);
        }

        // Store current page instance for potential use
        window.currentPageInstance = pageInstance;

        // Store for dataset detail page
        if (pageInstance.constructor.name === 'DatasetDetailPage') {
            window.currentDatasetDetailPage = pageInstance;
        }

        // Store for training page
        if (pageInstance.constructor.name === 'TrainingPage') {
            window.trainingPage = pageInstance;
        }

        // Store for evaluation page
        if (pageInstance.constructor.name === 'EvaluationPage') {
            window.evaluationPage = pageInstance;
        }
    }

    updateActiveNav(currentPath) {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === `#${currentPath}`) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    setupNavigation() {
        // Handle all nav clicks
        document.addEventListener('click', (e) => {
            if (e.target.matches('a[href^="#"]')) {
                const navbarCollapse = document.getElementById('navbarNav');
                if (navbarCollapse && navbarCollapse.classList.contains('show')) {
                    const bsCollapse = new bootstrap.Collapse(navbarCollapse, {
                        toggle: true
                    });
                }
            }
        });
    }
}

// Not Found Page
class NotFoundPage {
    render() {
        return `
            <div class="container py-5">
                <div class="row justify-content-center">
                    <div class="col-md-6 text-center">
                        <h1 class="display-1 fw-bold text-primary">404</h1>
                        <h2 class="mb-4">Page Not Found</h2>
                        <p class="text-muted mb-4">The page you are looking for doesn't exist.</p>
                        <a href="#/" class="btn btn-primary">
                            <i class="bi bi-house me-2"></i>Go Home
                        </a>
                    </div>
                </div>
            </div>
        `;
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    new App();
});
