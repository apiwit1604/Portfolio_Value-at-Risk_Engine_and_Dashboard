# Portfolio Value at Risk (VaR) Engine

*[Read in English](README.md)*

คำนวณ VaR ของพอร์ตด้วย 3 วิธี — **Parametric (Variance-Covariance)**,
**Historical**, และ **Monte Carlo** — หา portfolio weights ที่ minimize risk
หรือ minimize tail loss ผ่าน constrained optimization และรัน rolling 1-year
backtest (Kupiec POF test) เพื่อประเมินว่าแต่ละโมเดล perform เป็นอย่างไรตาม
ช่วงเวลา

## โครงสร้าง

```
var_risk_engine/
├── src/
│   ├── config.py            # ค่าคงที่ทั้งหมด (tickers, dates, weights, horizon, ฯลฯ)
│   ├── data.py               # load_close_prices, compute_returns
│   ├── portfolio_stats.py    # scale_mu, scale_cov (scaling ตาม horizon)
│   ├── var_models.py         # performance_portfolio, historical_var, monte_carlo_var
│   ├── optimize.py           # find_min_risk_portfolio, find_min_loss_portfolio
│   ├── backtest.py           # rolling windows, calculate_var_metrics, kupiec_test
│   └── report.py             # show_portfolio, show_var_comparison
├── notebooks/
│   └── var_risk_engine.ipynb # notebook เล่าเรื่อง เรียกใช้ฟังก์ชันจาก src/
├── requirements.txt
├── README.md                 # ไฟล์นี้เวอร์ชันภาษาอังกฤษ
└── README.th.md              # ไฟล์นี้ (ภาษาไทย)
```

Logic การคำนวณทั้งหมดอยู่ใน `src/` ส่วน notebook เป็นแค่ชั้นเล่าเรื่อง/แสดงผล
การแยกแบบนี้ทำให้ logic สามารถ unit-test หรือเอาไปใช้ใน script อื่นได้โดยไม่
ต้องพึ่ง notebook และ notebook เองก็อ่านง่ายขึ้น เพราะไม่ปนกันระหว่างรายละเอียด
การ implement กับคำอธิบาย

## การติดตั้ง

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/var_risk_engine.ipynb
```

## ข้อจำกัดที่ต้องรู้ก่อนเชื่อผลลัพธ์จาก backtest

`src/backtest.py` มี **look-ahead bias ที่ระบุไว้ชัดเจนในโค้ด**: rolling
window ที่ใช้ *ประมาณ* VaR ของแต่ละวัน ปัจจุบันรวมข้อมูลของวันเดียวกับที่เอา
ไปใช้ *เช็คผล* แทนที่จะหยุดที่วันก่อนหน้า พูดง่ายๆ คือโมเดล "เห็น" ผลลัพธ์ที่
มันกำลังถูกเอาไปเทียบใน Kupiec test บางส่วน ซึ่งมีแนวโน้มทำให้ทั้ง 3 โมเดลดู
แม่นยำกว่าความเป็นจริง

จุดนี้ **ยังไม่ได้แก้** ในเวอร์ชันปัจจุบัน — ดู docstring ด้านบนของ
`src/backtest.py` เพื่อดูกลไก indexing ที่แน่นอนและวิธีแก้ (แก้ได้ในบรรทัด
เดียว) ให้มองผลลัพธ์ใน Section 11-12 ของ notebook เป็นแค่ตัวอย่าง workflow
ไม่ใช่ผลการวัด performance ที่ validate แล้ว จนกว่าจะแก้จุดนี้

ข้อจำกัดรองอีกจุด: ที่ `N_ROLLING_WINDOWS = 252` และ confidence 95%
จำนวน exception ที่คาดหวังมีแค่ราว 12-13 ครั้ง ทำให้ Kupiec test มี
statistical power ต่ำในขนาด sample นี้ — ผลที่ออกมาว่า "Pass" ถือเป็น
หลักฐานที่ค่อนข้างอ่อน ไม่ใช่บทพิสูจน์ว่าโมเดล adequate

## สรุปวิธีการ

| วิธี | สมมติฐาน | แนวทาง |
|---|---|---|
| Parametric | Return มีการแจกแจงแบบปกติ (normal) | `return + z * risk` โดย z มาจาก normal quantile |
| Historical | ไม่มีสมมติฐานเรื่องการแจกแจง | Empirical quantile ของ return จริงของพอร์ต |
| Monte Carlo | GBM พร้อม correlated shocks | จำลองด้วย Cholesky-correlated simulation แล้วหา quantile ของพอร์ตที่จำลองได้ |

VaR ถูกเก็บเป็นค่า **ลบ** ตลอดทั้งโปรเจกต์ (ยิ่งติดลบมาก = loss ยิ่งแย่) —
จุดนี้มีผลต่อการอ่าน `find_min_loss_portfolio`: ฟังก์ชันนี้ *maximize* VaR
(ดันค่าเข้าใกล้ 0) ซึ่งเทียบเท่ากับการ minimize ขนาดของ tail loss
