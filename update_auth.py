with open('auth_router.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('router   = APIRouter(prefix="/auth", tags=["auth"])', 'router   = APIRouter(tags=["auth"])')
text = text.replace('@router.post("/register")', '@router.post("/auth/register")')
text = text.replace('@router.post("/send-otp")', '@router.post("/auth/send-otp")')
text = text.replace('@router.post("/verify-otp")', '@router.post("/auth/verify-otp")')
text = text.replace('@router.post("/login")', '@router.post("/auth/login")')
text = text.replace('@router.post("/reset-password")', '@router.post("/auth/reset-password")')
text = text.replace('@router.get("/me")', '@router.get("/auth/me")')

with open('auth_router.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated auth_router.py')
