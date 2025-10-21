# VisionAI Platform - Frontend

## 🚀 빠른 시작

### 1. Firebase 설정 (필수)

Firebase 인증을 사용하기 위해 다음 단계를 따르세요:

#### Step 1: Firebase 프로젝트 생성
1. [Firebase Console](https://console.firebase.google.com/) 접속
2. "프로젝트 추가" 클릭
3. 프로젝트 이름 입력 후 생성

#### Step 2: Authentication 활성화
1. 왼쪽 메뉴 > **Build > Authentication**
2. "시작하기" 클릭
3. **Sign-in method** 탭에서 활성화:
   - ✅ **Email/Password** (필수)
   - ✅ **Google** (선택사항)

#### Step 3: 웹 앱 등록
1. 프로젝트 개요 > "앱 추가" > 웹 아이콘(`</>`) 클릭
2. 앱 닉네임 입력 > "앱 등록"
3. **Firebase SDK 구성 정보 복사**

#### Step 4: 프로젝트 설정 파일 수정
`FE/js/config/firebase-config.js` 파일을 열고 복사한 정보로 교체:

```javascript
const firebaseConfig = {
    apiKey: "여기에_실제_API_KEY",
    authDomain: "여기에_실제_AUTH_DOMAIN",
    projectId: "여기에_실제_PROJECT_ID",
    storageBucket: "여기에_실제_STORAGE_BUCKET",
    messagingSenderId: "여기에_실제_SENDER_ID",
    appId: "여기에_실제_APP_ID"
};
```

**⚠️ 중요**: `YOUR_API_KEY_HERE` 같은 플레이스홀더를 실제 값으로 교체하세요!

### 2. 백엔드 서버 실행

```bash
cd BE
uvicorn main:app --reload
```

서버가 `http://localhost:8000`에서 실행됩니다.

### 3. 프론트엔드 실행

#### 옵션 A: Live Server (추천)
1. VS Code에서 `index.html` 우클릭
2. "Open with Live Server" 선택

#### 옵션 B: Python HTTP Server
```bash
cd FE
python -m http.server 8080
```
`http://localhost:8080`에서 접속

#### 옵션 C: 브라우저로 직접 열기
`FE/index.html` 파일을 더블클릭하여 브라우저로 열기

---

## ✅ 설정 확인

### 브라우저 콘솔 확인 (F12)
올바르게 설정되었다면 다음 메시지가 표시됩니다:
```
Firebase initialized successfully
```

### 에러 메시지별 해결 방법

#### ❌ "auth/api-key-not-valid"
→ `firebase-config.js`에서 API 키를 실제 Firebase 값으로 교체하세요.

#### ❌ "Firebase config not found"
→ `index.html`에서 `firebase-config.js` 스크립트가 로드되는지 확인하세요.

#### ❌ "Firebase SDK not loaded"
→ 인터넷 연결을 확인하세요 (Firebase CDN 접근 필요).

---

## 🧪 테스트

### 1. 회원가입 테스트
1. 우측 상단 **"Sign Up"** 클릭
2. 정보 입력:
   - Display Name: Test User
   - Email: test@example.com
   - Password: test123456
3. "Sign Up" 클릭
4. ✅ 성공 시 우측 상단에 사용자 이름 표시

### 2. Firebase Console에서 확인
1. [Firebase Console](https://console.firebase.google.com/) > Authentication > Users
2. 생성된 사용자 확인

---

## 📁 프로젝트 구조

```
FE/
├── index.html                      # 메인 HTML
├── css/
│   └── style.css                   # 스타일시트
├── js/
│   ├── app.js                      # 라우터
│   ├── config/
│   │   └── firebase-config.js      # 🔥 Firebase 설정 (수정 필요!)
│   ├── services/
│   │   ├── api.js                  # Backend API 서비스
│   │   └── firebase.js             # Firebase Auth 서비스
│   └── pages/
│       ├── home.js                 # 홈 페이지
│       ├── datasets.js             # 데이터셋 관리
│       ├── auto-annotate.js        # 자동 어노테이션
│       ├── dataset-detail.js       # 데이터셋 상세
│       ├── training.js             # 모델 학습
│       ├── conversion.js           # 모델 변환
│       ├── evaluation.js           # 모델 평가
│       ├── deployment.js           # 모델 배포
│       └── monitoring.js           # 모니터링
```

---

## 🔧 주요 기능

### ✅ 구현 완료
- [x] Firebase 이메일/비밀번호 인증
- [x] Firebase Google 로그인
- [x] 데이터셋 업로드 및 관리
- [x] 데이터셋 설정 모달
- [x] 자동 어노테이션 페이지
- [x] 데이터셋 상세 뷰어
- [x] 모델 학습 시작 모달
- [x] 학습 작업 관리 및 모니터링
- [x] 실시간 메트릭 차트

### ⏳ 개발 중
- [ ] 모델 변환 기능
- [ ] 모델 평가 메트릭 표시
- [ ] 모델 배포 기능
- [ ] 모니터링 및 재학습 트리거

---

## 🔒 보안 참고사항

### Firebase API 키
- ✅ 클라이언트에 노출되어도 안전 (Firebase 설계 상)
- ⚠️ `.gitignore`에 민감한 백엔드 키는 추가하세요
- 🔒 Firebase Console에서 승인된 도메인만 허용

### 권장사항
- HTTPS 사용 (배포 시)
- Firebase Security Rules 설정
- 환경 변수로 민감한 정보 관리 (백엔드)

---

## 📚 추가 문서

자세한 Firebase 설정 가이드는 [`FIREBASE_SETUP_GUIDE.md`](../FIREBASE_SETUP_GUIDE.md)를 참조하세요.

---

## 🐛 문제 해결

### 로그인/회원가입 모달이 중복으로 표시됨
→ **해결됨**: 최신 코드에서 자동으로 이전 모달 제거

### API 호출 실패 (CORS 에러)
→ 백엔드 서버가 실행 중인지 확인 (`http://localhost:8000`)

### 차트가 표시되지 않음
→ Chart.js CDN이 로드되었는지 확인 (개발자 도구 > Network)

---

## 📞 지원

문제가 발생하면:
1. 브라우저 콘솔(F12) 확인
2. `FIREBASE_SETUP_GUIDE.md` 문서 참조
3. Firebase Console에서 설정 재확인

---

**마지막 업데이트**: 2025-10-16
